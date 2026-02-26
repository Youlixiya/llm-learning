<div align="center">

# 🚀 LLM 实战教程

**从 API 调用到生产级应用的全栈 LLM 工程实践**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://youlixiya.github.io/llm-learning/)

[在线文档](https://youlixiya.github.io/llm-learning/) · [快速开始](#-快速开始) · [学习路线](#-学习路线) · [项目结构](#-项目结构)

</div>

---

## 📖 项目简介

本项目是一套**面向工程实践**的现代大语言模型（LLM）学习与实战教程，使用 **Python + PyTorch** 构建。面向已有编程基础的开发者，帮助你从"会调用接口"进阶到"能设计和实现完整 LLM 应用"。

### ✨ 核心特性

- 🎯 **工程导向**：每一章都配套可运行的代码与脚本，优先解决"能跑起来"和"能落地"的问题
- 📈 **循序渐进**：从 Tiny Transformer 玩具模型，到指令微调（SFT + LoRA）、RAG、Agent、评测和部署
- 🇨🇳 **中文讲解**：所有文档以中文撰写，配合必要的英文术语，方便查阅官方资料
- 🔧 **可复用代码**：`src/` 下的模块可以直接在你的真实项目中引用和改造
- 📚 **完整示例**：涵盖数据准备、模型训练、微调、RAG 构建、Agent 开发、API 部署等全流程

---

## 🎓 学习路线

### 阶段一：基础理解（Tiny LM）

1. **Transformer 基础** (`docs/01_transformer_basics.md`)
   - 理解"预测下一个 token"的核心思想
   - 从零实现 Tiny Transformer
   - 掌握注意力机制、位置编码等关键概念

2. **Tiny LM 训练实践** (`docs/02_tiny_lm_training.md`)
   - 数据准备与预处理
   - 训练流程与超参数调优
   - 文本生成与评估

### 阶段二：实用技术（微调与增强）

3. **指令微调与 LoRA** (`docs/03_instruction_finetune_lora.md`)
   - 参数高效微调（PEFT）方法
   - LoRA / QLoRA 实战
   - 在开源模型上做指令微调

4. **RAG 系统构建** (`docs/04_rag_system.md`)
   - 向量数据库与检索
   - 知识库构建与查询
   - 检索增强生成完整流程

### 阶段三：高级应用（Agent 与部署）

5. **Agent 与工具调用** (`docs/05_agents_and_tools.md`)
   - 工具调用机制
   - 多轮决策与推理
   - 复杂任务编排

6. **评测与部署**（规划中）
   - 模型评测体系
   - FastAPI 服务部署
   - 生产环境最佳实践

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.10 或更高版本
- **PyTorch**: 2.2.0+
- **CUDA**: 可选，用于 GPU 加速训练

### 安装步骤

1. **克隆仓库**

```bash
git clone https://github.com/Youlixiya/llm-learning.git
cd llm-learning
```

2. **创建虚拟环境**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. **安装依赖**

```bash
pip install -U pip
pip install -r requirements.txt
```

4. **运行第一个示例**

```bash
# 训练 Tiny Transformer 并生成文本
bash scripts/run_tiny_lm.sh
```

### 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## 📁 项目结构

```
llm-learning/
├── docs/                    # 📚 教程文档（Markdown）
│   ├── 00_intro.md         # 引言与整体规划
│   ├── 01_transformer_basics.md
│   ├── 02_tiny_lm_training.md
│   ├── 03_instruction_finetune_lora.md
│   ├── 04_rag_system.md
│   └── 05_agents_and_tools.md
│
├── src/                     # 🔧 可复用的工程代码
│   ├── tiny_lm/            # Tiny Transformer 语言模型
│   │   ├── model.py        # 模型定义
│   │   ├── train.py        # 训练脚本
│   │   └── generate.py     # 文本生成
│   ├── finetune/           # 指令微调 & LoRA
│   │   ├── train_lora.py
│   │   └── infer_lora.py
│   ├── rag/                # 检索增强生成（RAG）
│   │   ├── build_index.py  # 构建向量索引
│   │   └── query_rag.py    # RAG 查询
│   ├── agents/             # Agent 与工具调用
│   │   └── simple_agent.py
│   ├── evals/              # 评测脚本
│   └── api/                # API 服务（FastAPI）
│
├── scripts/                 # 🚀 一键运行脚本
│   ├── run_tiny_lm.sh
│   ├── run_finetune_lora.sh
│   ├── run_rag_demo.sh
│   └── run_agent_demo.sh
│
├── data/                    # 📊 示例数据
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后数据
│
├── notebooks/               # 📓 Jupyter Notebooks
│
├── web/                     # 🌐 GitHub Pages 网站
│   ├── index.html
│   ├── chapters/           # 章节 HTML
│   └── docs/               # 文档副本
│
├── requirements.txt         # Python 依赖
└── README.md               # 本文件
```

---

## 🛠️ 核心模块说明

### Tiny LM (`src/tiny_lm/`)

最小可运行的 Transformer 语言模型实现，用于理解 LLM 的核心机制，并在此基础上进一步做指令 SFT 和多轮对话。

```bash
# 1）从零训练一个字符级 Tiny LM
python src/tiny_lm/train.py

# 使用训练好的模型做文本生成
python src/tiny_lm/generate.py --prompt "今天天气"

# 2）基于 Qwen 分词器进行 SFT，并导出 HF 适配权重
python src/tiny_lm/train_sft.py

# 3）加载 SFT 后的 TinyLM，启动本地多轮对话

# 命令行对话（CLI）
python src/tiny_lm/chat_ui.py --mode cli

# Web 多轮对话（Gradio）
python src/tiny_lm/chat_ui.py --mode gradio --port 7860
```

> 说明：`train_sft.py` 会在 `data/processed/` 下生成基于 Qwen 分词器的 TinyLM HF checkpoint，  
> `chat_ui.py` 复用同一套 chat 模板，既可以在终端中对话，也可以通过浏览器访问 Gradio WebUI 进行多轮对话。

### 指令微调 (`src/finetune/`)

基于 Hugging Face Transformers 和 PEFT 的参数高效微调实现。

```bash
# LoRA 微调
bash scripts/run_finetune_lora.sh
```

### RAG 系统 (`src/rag/`)

完整的检索增强生成系统，支持向量数据库和语义检索。

```bash
# 构建索引并查询
bash scripts/run_rag_demo.sh
```

### Agent (`src/agents/`)

简单的 Agent 实现，支持工具调用和多轮对话。

```bash
# 运行 Agent 示例
bash scripts/run_agent_demo.sh
```

---

## 📚 在线文档

访问 [GitHub Pages](https://youlixiya.github.io/llm-learning/) 查看完整的在线教程文档。

文档包含：
- 📖 交互式章节导航
- 💡 代码示例与说明
- 🔍 搜索功能
- 🌓 深色/浅色主题切换

---

## 🤝 贡献指南

欢迎贡献代码、文档或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📝 许可证

本项目采用 [MIT License](LICENSE) 许可证。

---

## 🙏 致谢

- [Hugging Face Transformers](https://github.com/huggingface/transformers) - 模型库与工具
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [PEFT](https://github.com/huggingface/peft) - 参数高效微调

---

## 📮 联系方式

- **GitHub**: [@Youlixiya](https://github.com/Youlixiya)
- **Issues**: [提交问题](https://github.com/Youlixiya/llm-learning/issues)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star！**

Made with ❤️ by [youlixiya](https://github.com/Youlixiya)

</div>