import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_RAG_DIR = DATA_DIR / "raw" / "rag_corpus"


@dataclass
class Tool:
    name: str
    description: str
    schema: Dict[str, Any]
    func: Callable[..., str]


def tool_search_docs(query: str) -> str:
    """在本地 RAG 语料目录中做最简单的关键词搜索。"""
    if not RAW_RAG_DIR.exists():
        return f"文档目录 {RAW_RAG_DIR} 不存在，无法搜索。"

    hits: List[str] = []
    for path in sorted(RAW_RAG_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if query in text:
            snippet = text[:400].replace("\n", " ")
            hits.append(f"文件: {path.name}\n片段: {snippet}")

    if not hits:
        return f"在文档目录中没有找到包含关键词“{query}”的内容。"

    return "\n\n".join(hits[:5])


def tool_calculator(expression: str) -> str:
    """执行简单的数学表达式计算。仅支持 + - * / 和括号。"""
    # 非严格但相对安全的白名单校验
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return "表达式包含非法字符，仅支持数字、加减乘除和括号。"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as e:  # noqa: BLE001
        return f"计算出错：{e}"
    return f"{expression} = {result}"


def tool_get_time() -> str:
    """返回当前时间。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"当前时间：{now}"


def get_available_tools() -> Dict[str, Tool]:
    return {
        "search_docs": Tool(
            name="search_docs",
            description="在本地教程文档中按关键词搜索相关片段，用于回答与本仓库内容相关的问题。",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要搜索的中文关键词或短句"},
                },
                "required": ["query"],
            },
            func=lambda **kwargs: tool_search_docs(kwargs["query"]),
        ),
        "calculator": Tool(
            name="calculator",
            description="计算一个简单的数学表达式，仅支持 + - * / 和括号。",
            schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如 (3 + 5) * 12 / 4",
                    },
                },
                "required": ["expression"],
            },
            func=lambda **kwargs: tool_calculator(kwargs["expression"]),
        ),
        "get_time": Tool(
            name="get_time",
            description="获取当前日期和时间。",
            schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
            func=lambda **_: tool_get_time(),
        ),
    }


def build_system_prompt(tools: Dict[str, Tool]) -> str:
    tool_descs = []
    for t in tools.values():
        tool_descs.append(
            f"- 工具名: {t.name}\n  功能: {t.description}\n  参数 JSON Schema: {json.dumps(t.schema, ensure_ascii=False)}"
        )
    tools_block = "\n\n".join(tool_descs)

    prompt = f"""
你是一个可以调用工具的中文助手。你可以使用以下工具来辅助完成任务：

{tools_block}

当你觉得需要调用工具时，请严格输出 JSON 格式：
{{"type": "tool_call", "tool": "<工具名>", "arguments": {{...}}}}

当你已经有足够信息，可以直接回答用户问题时，请输出：
{{"type": "final_answer", "answer": "你的最终回答"}}

不要输出任何额外内容，不要包含解释或自然语言，只输出一个合法的 JSON 对象。
"""
    return prompt.strip()


def load_model_and_tokenizer(model_name_or_path: str):
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
    model.eval()
    return model, tokenizer


def call_llm(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
) -> str:
    device = next(model.parameters()).device
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
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0, inputs.shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def extract_json(text: str) -> Dict[str, Any]:
    """从模型输出中提取第一个 JSON 对象。"""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"未能在模型输出中找到 JSON：{text}")
    return json.loads(match.group(0))


def run_agent(model_name_or_path: str) -> None:
    tools = get_available_tools()
    system_prompt = build_system_prompt(tools)

    print("加载模型中，请稍候...")
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)

    print("Agent 已就绪。输入你的指令（输入 exit 退出）。")
    while True:
        user_input = input("输入你的指令> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        # 第一步：让模型决定是否调用工具，或直接回答
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        raw = call_llm(model, tokenizer, messages)
        try:
            decision = extract_json(raw)
        except Exception as e:  # noqa: BLE001
            print(f"[解析错误] {e}")
            print(f"原始输出：{raw}")
            continue

        if decision.get("type") == "final_answer":
            print(f"\n助手：{decision.get('answer', '').strip()}\n")
            continue

        if decision.get("type") != "tool_call":
            print(f"[协议错误] 非预期的 type 字段：{decision}")
            continue

        tool_name = decision.get("tool")
        arguments = decision.get("arguments", {}) or {}

        if tool_name not in tools:
            print(f"[错误] 未知工具：{tool_name}")
            continue

        tool = tools[tool_name]
        try:
            tool_result = tool.func(**arguments)
        except TypeError as e:
            print(f"[工具调用错误] 参数不匹配：{e}")
            continue

        # 第二步：把工具结果反馈给模型，让它给出最终回答
        followup_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {
                "role": "assistant",
                "content": json.dumps(decision, ensure_ascii=False),
            },
            {
                "role": "user",
                "content": (
                    f"工具 {tool_name} 已被调用，返回结果如下：\n{tool_result}\n\n"
                    "请根据这个结果给出最终回答，并严格以 "
                    '{"type": "final_answer", "answer": "..."} 这样的 JSON 对象形式返回。'
                ),
            },
        ]

        raw2 = call_llm(model, tokenizer, followup_messages)
        try:
            final = extract_json(raw2)
            print(f"\n助手：{final.get('answer', '').strip()}\n")
        except Exception as e:  # noqa: BLE001
            print(f"[解析错误] {e}")
            print(f"原始输出：{raw2}")


def main() -> None:
    parser = argparse.ArgumentParser(description="最小可运行的 LLM Agent 示例（工具调用）")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="用于驱动 Agent 的 Chat 模型名称或路径",
    )
    args = parser.parse_args()
    run_agent(args.model_name_or_path)


if __name__ == "__main__":
    main()

