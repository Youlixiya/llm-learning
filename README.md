## LLM 实战教程（Python + PyTorch）

本项目是一套面向 **工程实践** 的现代大语言模型（LLM）学习与实战教程，使用 **Python + PyTorch**，面向已经有一定编程基础的开发者，目标是帮助你从“会调用接口”进阶到“能设计和实现完整 LLM 应用”。

### 教程特点

- **工程导向**：每一章都配套可运行的代码与脚本，优先解决“能跑起来”和“能落地”的问题。
- **从浅到深**：从 Tiny Transformer 玩具模型，到指令微调（SFT + LoRA）、RAG、Agent、评测和部署。
- **中文讲解**：所有文档以中文撰写，配合必要的英文术语，方便查阅官方资料。

### 目录结构概览

```text
docs/         # 教程文档（中文）
notebooks/    # 交互式实验 Notebook
src/          # 可复用的工程代码
  tiny_lm/    # Tiny Transformer 语言模型示例
  finetune/   # 指令微调、LoRA/QLoRA 相关代码
  rag/        # RAG（检索增强生成）系统
  agents/     # Agent 与工具调用
  evals/      # 评测脚本
  api/        # API 服务（FastAPI）
data/         # 示例数据（原始与处理后）
scripts/      # 一键运行脚本
```

### 起步：环境准备

1. 创建并激活虚拟环境（推荐 Python 3.10+）：

```bash
cd "llm learning"
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
```

2. 安装依赖：

```bash
pip install -U pip
pip install -r requirements.txt
```

3. 运行第一个示例（Tiny Transformer 训练与生成）：

```bash
bash scripts/run_tiny_lm.sh
```

> 详细原理请参考 `docs/01_transformer_basics.md` 与 `docs/02_tiny_lm_training.md`。

### 学习路线建议

1. **从 Tiny LM 入门**：理解“语言模型 = 预测下一个 token”的核心思想，跑通 `src/tiny_lm`。
2. **参数高效微调（PEFT）**：在开源模型上做指令微调，快速打造自己的专用助手。
3. **构建知识库问答（RAG）**：让模型学会“先查资料，再回答”，解决闭卷知识缺失问题。
4. **Agent 与工具调用**：让模型学会调用搜索、数据库、计算等工具，完成更复杂任务。
5. **评测与部署**：为自己的 LLM 系统设计统一的评测流程，并以 API 形式对外提供服务。

后续章节会逐步完善示例代码与文档，你可以按需从任意模块切入，也可以从头到尾完整学习。

