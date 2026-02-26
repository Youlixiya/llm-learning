## 3. 用 LoRA 做指令微调：把通用大模型变成“你的专用助手”

这一章我们从工程视角介绍：如何在一个开源大模型上做 **指令微调（Instruction Tuning）**，并通过 **LoRA** 这种参数高效微调方法，将模型变成适合自己场景的“专用助手”。

> 目标依然是“最小可运行版本”：  
> **准备一个小的指令数据集 → 用 LoRA 微调开源模型 → 用微调后的模型进行对话。**

---

### 3.1 什么是指令微调（Instruction Tuning）？

从第 1、2 章你已经知道：

- 语言模型本质是在做：**给定前文，预测下一个 token**；
- 经过大量预训练后，模型学会了丰富的语言模式和世界知识。

但是，**预训练模型并不知道“你到底想让它做什么”**：

- 你说“总结一下这段话”，它可能只是续写；
- 你说“帮我写一份周报”，它不一定知道要用什么格式；
- 你给它一段 JSON，它也未必知道是要“解析”还是“翻译”。

**指令微调** 的目的就是：

> 让模型学会：**看到一条“指令”，就按照我们期望的方式去理解和回答。**

典型的指令数据长这样：

```json
{"instruction": "用三句话总结下面的文本", "input": "……一段长文本……", "output": "……总结结果……"}
{"instruction": "把下面的接口文档转成 Markdown 表格", "input": "{...JSON...}", "output": "……Markdown 表格……"}
{"instruction": "你现在是一个面试官，生成 5 个关于 Python 并发的面试题", "input": "", "output": "……5 个问题……"}
```

模型通过在大量这样的样本上继续训练，学会对“自然语言指令 + 输入”做出合适的响应。

---

### 3.2 为什么要用 LoRA？（参数高效微调）

直接在一个 7B、13B 甚至更大的模型上“全参数微调”几乎是不现实的：

- 显存占用极大（几十 GB 起步）；
- 训练更新的参数量和存储空间都很大；
- 对个人开发者和小团队非常不友好。

**LoRA（Low-Rank Adaptation）** 提供了一条工程上非常实用的道路：

- 不再更新原模型中所有的权重矩阵；
- 只在部分线性层上，添加一个小的 **低秩适配器（低维矩阵分解）**；
- 训练时只更新这些小矩阵的参数；
- 推理时仍然使用原模型 + 适配器的合成效果。

直觉上：

> 你可以把 LoRA 理解成“在原乐谱上加一些轻量的和声记号”，  
> 原曲不变，但在特定场景下可以弹出不一样的风格。

好处：

- **显存和算力开销大幅降低**，可以在消费级显卡甚至 CPU 上做小规模实验；
- **参数可插拔**：可以为不同任务训练多个 LoRA 适配器，在同一个底座模型上切换；
- **方便分享和部署**：只需发布小体积的适配器权重即可。

在本教程中，我们使用 Hugging Face 的 `peft` 库来实现 LoRA。

---

### 3.3 代码结构与运行方式

本章相关代码位于：

- `src/finetune/train_lora.py`：使用 LoRA 在指令数据上进行微调；
- `src/finetune/infer_lora.py`：加载微调好的 LoRA 适配器进行对话推理。

我们默认使用一个支持中文的开源指令模型（例如 `Qwen/Qwen2.5-0.5B-Instruct`），你也可以通过命令行参数替换为任意兼容的 Chat 模型。

#### 3.3.1 准备环境

确保已经安装完本项目依赖，并推荐使用 Python 虚拟环境（参见 `README.md`）：

```bash
pip install -U pip
pip install -r requirements.txt
```

> 如果你计划在 GPU 上训练，请确保正确安装了带 CUDA 的 PyTorch，并在 `nvidia-smi` 中能看到显卡。

#### 3.3.2 运行一个最小 LoRA 微调示例

`train_lora.py` 内置了一小批示例指令数据（中文为主），方便你不开新数据也能跑通流程。

在项目根目录下运行：

```bash
python src/finetune/train_lora.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir data/processed/finetune/qwen25-0_5b-lora \
  --max_steps 50
```

这会做几件事：

1. 从 Hugging Face 下载并加载基座模型和分词器；
2. 使用内置的小指令数据构造训练集；
3. 为模型添加 LoRA 适配器，只训练这些小模块的参数；
4. 将训练得到的 LoRA 适配器权重保存到 `output_dir`。

训练完成后，你可以用 `infer_lora.py` 来测试微调效果：

```bash
python src/finetune/infer_lora.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_dir data/processed/finetune/qwen25-0_5b-lora \
  --prompt "请用 3 条要点总结：为什么要学习大语言模型工程？"
```

如果一切顺利，你会看到模型的回答更贴近我们在指令数据中展示的“说话风格”和“任务偏好”。

---

### 3.4 数据格式与自定义指令集

在真实项目里，你肯定不会只用内置的几条玩具数据。  
`train_lora.py` 支持从一个 JSONL 文件中加载自定义指令数据，格式约定如下：

```json
{"instruction": "指令内容", "input": "可选的补充输入", "output": "期望的回答"}
{"instruction": "请用要点列出下面 API 的入参", "input": "{...接口 JSON...}", "output": "- 参数 a: 含义 ..."}
{"instruction": "你现在是一个 HR，帮我写一封面试邀请邮件", "input": "", "output": "……邮件正文……"}
```

保存为例如 `data/raw/instruct_tiny.jsonl`，每行一条 JSON。

运行微调时指定：

```bash
python src/finetune/train_lora.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_path data/raw/instruct_tiny.jsonl \
  --output_dir data/processed/finetune/my-lora \
  --max_steps 100
```

脚本会：

- 自动解析 JSONL 中的字段（`instruction` / `input` / `output`）；
- 使用底座模型的 `chat_template` 构造对话格式；
- 将整个“指令 → 回答”序列作为监督信号进行 SFT（Supervised Fine-Tuning）。

> 提示：  
> 数据量很小时（几百条以内），不要指望模型能力有质变。这更多是一个“定制说话风格”和“对齐业务语境”的过程。

---

### 3.5 一些工程注意事项

- **选择合适的基座模型**
  - 如果主要是中文场景，可以选 `Qwen` 系列、`Yi` 系列等中文友好的模型；
  - 如果是中英文混合，`Qwen` / `Llama` 系列的 Instruct 模型也表现不错。
- **LoRA 配置**
  - 为了简单，我们在示例里使用了较小的秩（例如 `r=8`），适合快速实验；
  - 在你的项目中，可以通过调整 `r`、`lora_alpha`、`target_modules` 等参数进行权衡。
- **训练资源**
  - 即便使用 LoRA，完整微调一个 7B 模型在 CPU 上仍然非常吃力；
  - 推荐在至少 1 块 12GB 以上显存的 GPU 上进行实验，或者选择更小的基座模型。

---

### 3.6 本章小结与下一步

到这里，你已经掌握了一个最常用的工程套路：

- 明白了什么是指令微调，以及它与普通预训练的区别；
- 知道了 LoRA 这种参数高效微调方法的直觉和优势；
- 能够在本项目中使用 `train_lora.py` 和 `infer_lora.py` 对开源模型做一次小规模指令微调。

在接下来的章节中，我们会在这个基础上进一步：

- 使用微调好的模型作为 **生成组件**，构建一个 **RAG 知识库问答系统**；
- 再往后，引入 **Agent 与工具调用** 的概念，让模型学会主动“查资料、算东西、调接口”。

