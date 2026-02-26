## 2. 从零训练一个 Tiny Transformer 语言模型

这一章我们会真正动起手来，从零训练一个 **迷你版 Transformer 语言模型**，并让它学会在一个小语料上生成文本。

> 目标不是追求效果有多好，而是让你完整跑一遍：  
> **数据准备 → 模型定义 → 训练循环 → 文本生成** 的工程流程。

---

### 2.1 代码结构与运行方式

本章相关代码位于：

- `src/tiny_lm/model.py`：Tiny Transformer 模型定义；
- `src/tiny_lm/train.py`：训练脚本（包含一个内置玩具语料）；
- `src/tiny_lm/generate.py`：加载已训练模型进行文本生成；
- `scripts/run_tiny_lm.sh`：一键运行训练 + 生成的脚本。

#### 运行示例

```bash
# 训练一个 Tiny LM（会在 data/processed/ 下保存模型和词表）
python src/tiny_lm/train.py

# 使用训练好的模型从提示语生成文本
python src/tiny_lm/generate.py --prompt "今天天气"

# 或者使用脚本一键运行
bash scripts/run_tiny_lm.sh
```

第一次运行时，训练会在一个极小的内置中文/英文混合玩具语料上进行，速度很快，只是为了演示流程。

---

### 2.2 数据准备：字符级玩具语料

为了让示例尽量“自带数据、开箱即用”，我们使用 **字符级语言模型**：

- 不依赖分词器或外部数据文件；
- 直接在代码里内置一小段文本作为训练语料；
- 使用“每个不同字符 = 一个 token”的方式构建词表。

在 `train.py` 里你会看到类似的流程：

1. 定义一个 `corpus` 字符串（几段简单的中英混合句子）；
2. 提取所有不同字符，构建 `stoi`（char → id）和 `itos`（id → char）；
3. 把整段文本编码为一长串 `ids` 列表；
4. 按固定窗口大小（例如 `block_size=64`）滑动切片，构建很多 `(input, target)` 样本：
   - `input` 是前 `block_size` 个 token；
   - `target` 是它们的“下一个 token”（整体右移一位）。

这种方式在真实项目中不会直接使用（通常会使用 BPE 或 SentencePiece 分词），但非常适合作为入门示例。

---

### 2.3 模型定义：TinyTransformerLM

`TinyTransformerLM` 是一个极简版的自回归 Transformer 语言模型，大致包含：

- `nn.Embedding`：将 token id 映射为向量；
- `nn.Embedding`：位置嵌入，向模型注入位置信息；
- `nn.TransformerEncoder`：由若干层 `nn.TransformerEncoderLayer` 堆叠而成；
- `nn.Linear`：把每个位置的隐藏状态映射回词表大小，得到对下一个 token 的 logits。

关键工程点：

- 使用 **因果 Mask（causal mask）**，保证模型在预测当前位置时只能看到之前的位置；
- 使用 `batch_first=True` 的 Transformer 编码器，简化张量维度的处理；
- 在前向传播中返回：
  - 对所有位置的 logits（训练用）；
  - 以及可选的最后一个位置 logits（生成时使用）。

你可以先把 `model.py` 当作“黑盒”，重点理解输入输出形状：

- 输入：`input_ids`，形状为 `(batch_size, seq_len)`；
- 输出：`logits`，形状为 `(batch_size, seq_len, vocab_size)`。

---

### 2.4 训练循环：从 loss 到可见的学习效果

在 `train.py` 中，我们实现了一个标准的训练循环：

1. 用 `DataLoader` 按批次提供 `(input_ids, target_ids)`；
2. 前向传播得到 `logits`；
3. 使用 `nn.CrossEntropyLoss` 计算损失：
   - 需要把 `logits` 和 `target_ids` reshape 成适合交叉熵的形状；
4. 反向传播 + `optimizer.step()` 更新参数；
5. 每隔若干 step 打印一次训练 loss，并用当前模型进行一次短文本生成，观察效果。

这里没有使用分布式训练、梯度累积等复杂技巧，目的是让你专注在“最小可运行版本”上。

> 在后面介绍使用大型开源模型和加速框架时，我们会再回到性能与稳定性的话题。

---

### 2.5 文本生成：采样策略与温度

在 `generate.py` 中，我们会：

1. 加载保存好的模型权重和 vocab（`tiny_lm.pt` 和 `tiny_lm_vocab.json`）；
2. 将输入的 `prompt` 文本编码为 token 序列；
3. 在循环中重复：
   - 截取最近的 `block_size` 个 token 作为模型输入；
   - 得到最后一个位置的 logits；
   - 根据采样策略（如 top-k 采样 + 温度）选择下一个 token；
   - 追加到已生成的序列中。

常见采样参数：

- **temperature**：温度，>1 会使分布更“均匀”，<1 更“保守”；
- **top_k**：仅在概率最高的前 k 个 token 中采样；
- **max_new_tokens**：最多生成多少个新 token。

虽然我们的 Tiny LM 非常小、语料也极少，但你仍然可以通过修改这些参数，感受到生成行为的差异。

---

### 2.6 本章小结与练习

到这里，你已经完成了一个最小闭环：

- 知道语言模型在做什么（预测下一个 token）；
- 理解一个简化版 Transformer 语言模型的结构；
- 能够从零训练并保存一个 Tiny LM；
- 能够用训练好的模型生成文本，并调整采样策略。

建议你尝试以下练习：

1. 修改训练语料，把它换成你自己的一小段文本（例如你写的几篇短文），观察模型是否会“学会你的语气”。
2. 调整模型尺寸（`d_model`、`num_layers`、`num_heads`），比较训练速度和生成效果的变化。
3. 修改 `block_size`（上下文长度），观察：
   - 太短时，模型记不住长依赖；
   - 太长时，训练速度和显存占用会如何变化。

在后续章节中，我们将从这个 Tiny LM 出发，过渡到使用成熟的开源大模型，并在它们之上进行指令微调和 RAG 系统的构建。

