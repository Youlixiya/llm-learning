import argparse
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAG_DB_DIR = PROCESSED_DIR / "rag_chroma"


def build_prompt(question: str, docs: List[Dict[str, Any]]) -> str:
    context_parts = []
    for i, d in enumerate(docs, start=1):
        context_parts.append(f"[资料 {i}] 来源: {d.get('source', 'unknown')}\n{d['text']}\n")
    context = "\n".join(context_parts)

    prompt = (
        "你是一个严谨的中文技术助手。下面是从知识库中检索到的资料片段，请严格基于这些资料回答用户问题。\n"
        "如果资料中没有相关信息，请直接说明“根据提供的资料无法回答”，不要编造。\n\n"
        f"{context}\n"
        f"用户问题：{question}\n"
        "请用中文给出简洁但有条理的回答："
    )
    return prompt


def load_rag_documents(
    db_dir: Path,
    embedding_model: str,
    question: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    if not db_dir.exists():
        raise FileNotFoundError(
            f"未找到 RAG 索引目录：{db_dir}，请先运行 build_index.py 构建索引。"
        )

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection("tutorial_rag")

    encoder = SentenceTransformer(embedding_model)
    query_emb = encoder.encode(
        [question],
        convert_to_numpy=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k,
    )

    docs: List[Dict[str, Any]] = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append({"text": text, "source": meta.get("source", "unknown")})
    return docs


def answer_with_rag(
    question: str,
    model_name_or_path: str,
    db_dir: Path,
    embedding_model: str,
    top_k: int = 3,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    docs = load_rag_documents(
        db_dir=db_dir,
        embedding_model=embedding_model,
        question=question,
        top_k=top_k,
    )
    if not docs:
        return "知识库中没有检索到相关文档。"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.to(device)
    model.eval()

    prompt = build_prompt(question, docs)

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": "你是一个严谨的中文技术助手，会基于提供的资料回答问题。",
        },
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0, inputs.shape[1] :]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    print("=== 检索到的资料来源 ===")
    for i, d in enumerate(docs, start=1):
        print(f"[{i}] {d['source']}")
    print("======================\n")

    return answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 RAG 知识库进行问答")
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="用户问题",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="用于生成答案的 Chat 模型",
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        default=str(RAG_DB_DIR),
        help="Chroma 持久化目录（需与 build_index.py 保持一致）",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-small-zh-v1.5",
        help="用于检索的向量模型名称",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="检索的文档数量",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="生成的最大新 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="nucleus sampling 的 top-p 参数",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    answer = answer_with_rag(
        question=args.question,
        model_name_or_path=args.model_name_or_path,
        db_dir=Path(args.db_dir),
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(answer)


if __name__ == "__main__":
    main()

