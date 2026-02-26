import argparse
from pathlib import Path
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_CORPUS_DIR = DATA_DIR / "raw" / "rag_corpus"
PROCESSED_DIR = DATA_DIR / "processed"
RAG_DB_DIR = PROCESSED_DIR / "rag_chroma"


def load_documents(source_dir: Path) -> List[dict]:
    """读取目录下的所有 Markdown 文档。"""
    if not source_dir.exists():
        raise FileNotFoundError(f"知识库目录不存在：{source_dir}")

    docs: List[dict] = []
    for path in sorted(source_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(
            {
                "id": path.stem,
                "text": text,
                "source": path.name,
            }
        )

    if not docs:
        raise ValueError(f"在目录 {source_dir} 中没有找到任何 .md 文档")

    return docs


def build_index(
    source_dir: Path,
    db_dir: Path,
    embedding_model: str,
) -> None:
    db_dir.mkdir(parents=True, exist_ok=True)

    print(f"从目录加载文档：{source_dir}")
    docs = load_documents(source_dir)
    print(f"共加载 {len(docs)} 篇文档。")

    print(f"加载向量模型：{embedding_model}")
    encoder = SentenceTransformer(embedding_model)

    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    metadatas = [{"source": d["source"]} for d in docs]

    print("编码文档为向量...")
    embeddings = encoder.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()

    print(f"写入 Chroma 向量库：{db_dir}")
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection("tutorial_rag")

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"已完成索引构建，共索引 {len(ids)} 条文本。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从本地文档构建 RAG 知识库索引")
    parser.add_argument(
        "--source_dir",
        type=str,
        default=str(RAW_CORPUS_DIR),
        help="知识库文档所在目录，默认使用 data/raw/rag_corpus",
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        default=str(RAG_DB_DIR),
        help="Chroma 持久化目录，默认使用 data/processed/rag_chroma",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-small-zh-v1.5",
        help="用于向量化文本的 SentenceTransformer 模型名称",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(
        source_dir=Path(args.source_dir),
        db_dir=Path(args.db_dir),
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()

