
我来用 **"hi, who are you"** 作为完整示例，通过严格的数学公式和文字描述，详细拆解 vLLM 的 Prefill 和 Decode 全过程。

---

## 一、基础设定

### 1.1 模型参数

| 参数 | 符号 | 数值（示例） |
|------|------|-------------|
| 词汇表大小 | $V$ | 32,000 |
| 隐藏层维度 | $d$ | 4,096 |
| 注意力头数 | $h$ | 32 |
| 每头维度 | $d_h = d/h$ | 128 |
| 层数 | $L$ | 32 |
| KV Cache 块大小 | $B$ | 16 tokens |
| 序列最大长度 | $T_{max}$ | 2,048 |

### 1.2 输入编码

输入字符串：**"hi, who are you"**

经过 Tokenizer 编码为 token ID 序列：
$$
\mathbf{x} = [x_1, x_2, x_3, x_4, x_5, x_6] = [\text{hi}, \text{,}, \text{who}, \text{are}, \text{you}, \text{?}]
$$

序列长度 $n = 6$。

---

## 二、Prefill 阶段（预填充阶段）

### 2.1 输入嵌入

将离散的 token ID 映射为连续向量：

$$
\mathbf{E} = \text{Embedding}(\mathbf{x}) \in \mathbb{R}^{n \times d}
$$

其中第 $i$ 个 token 的嵌入为 $\mathbf{e}_i \in \mathbb{R}^{d}$，因此：
$$
\mathbf{E} = \begin{bmatrix}
\mathbf{e}_1^\top \\
\mathbf{e}_2^\top \\
\vdots \\
\mathbf{e}_6^\top
\end{bmatrix} = \begin{bmatrix}
— & \mathbf{e}_1 & — \\
— & \mathbf{e}_2 & — \\
 & \vdots & \\
— & \mathbf{e}_6 & —
\end{bmatrix}_{6 \times 4096}
$$

加上位置编码 $\mathbf{P} \in \mathbb{R}^{n \times d}$：
$$
\mathbf{H}^{(0)} = \mathbf{E} + \mathbf{P}
$$

### 2.2 逐层 Transformer 计算（以第 $l$ 层为例）

#### 2.2.1 线性投影生成 Q、K、V

$$
\begin{aligned}
\mathbf{Q}^{(l)} &= \mathbf{H}^{(l-1)} \mathbf{W}_Q^{(l)\top} \in \mathbb{R}^{n \times d} \\
\mathbf{K}^{(l)} &= \mathbf{H}^{(l-1)} \mathbf{W}_K^{(l)\top} \in \mathbb{R}^{n \times d} \\
\mathbf{V}^{(l)} &= \mathbf{H}^{(l-1)} \mathbf{W}_V^{(l)\top} \in \mathbb{R}^{n \times d}
\end{aligned}
$$

其中权重矩阵 $\mathbf{W}_Q^{(l)}, \mathbf{W}_K^{(l)}, \mathbf{W}_V^{(l)} \in \mathbb{R}^{d \times d}$。

 reshape 为多头形式（$h=32$ 个头）：
$$
\mathbf{Q}^{(l)} = \begin{bmatrix} \mathbf{Q}_1^{(l)} & \mathbf{Q}_2^{(l)} & \cdots & \mathbf{Q}_h^{(l)} \end{bmatrix}, \quad \mathbf{Q}_j^{(l)} \in \mathbb{R}^{n \times d_h}
$$

#### 2.2.2 KV Cache 存储（PagedAttention 核心）

vLLM 将 $\mathbf{K}^{(l)}$ 和 $\mathbf{V}^{(l)}$ 分块存储到非连续的物理内存块中。

**块分配计算：**
$$
\text{所需块数} = \left\lceil \frac{n}{B} \right\rceil = \left\lceil \frac{6}{16} \right\rceil = 1 \text{ 个块}
$$

分配物理块 $\text{Block}_{phys}^{(l)}$（假设为系统空闲块 #42），建立**块表（Block Table）**：
$$
\mathcal{T}^{(l)} = [42]
$$

将 KV 张量写入块 #42：
$$
\begin{aligned}
\text{Block}_{42}.\mathbf{K} &= \mathbf{K}^{(l)} \in \mathbb{R}^{6 \times d} \quad (\text{实际存储为 } 6 \times 4096) \\
\text{Block}_{42}.\mathbf{V} &= \mathbf{V}^{(l)} \in \mathbb{R}^{6 \times d}
\end{aligned}
$$

注意：块容量为 16，当前只用了前 6 个位置，剩余 10 个位置预留给后续 decode 阶段追加。

#### 2.2.3 注意力计算（因果掩码）

对每个头 $j \in \{1, \dots, h\}$：

$$
\mathbf{S}_j = \frac{\mathbf{Q}_j^{(l)} \mathbf{K}_j^{(l)\top}}{\sqrt{d_h}} \in \mathbb{R}^{n \times n}
$$

应用因果掩码 $\mathbf{M} \in \{0, -\infty\}^{n \times n}$（下三角为 0，上三角为 $-\infty$）：
$$
\mathbf{S}_j^{masked} = \mathbf{S}_j + \mathbf{M}
$$

其中掩码矩阵：
$$
\mathbf{M}_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}
$$

Softmax 归一化：
$$
\mathbf{A}_j = \text{softmax}(\mathbf{S}_j^{masked}) \in \mathbb{R}^{n \times n}
$$

注意力输出：
$$
\mathbf{O}_j^{(l)} = \mathbf{A}_j \mathbf{V}_j^{(l)} \in \mathbb{R}^{n \times d_h}
$$

拼接所有头：
$$
\mathbf{O}^{(l)} = \text{Concat}[\mathbf{O}_1^{(l)}, \dots, \mathbf{O}_h^{(l)}] \mathbf{W}_O^{(l)\top} \in \mathbb{R}^{n \times d}
$$

#### 2.2.4 前馈网络与残差连接

$$
\begin{aligned}
\mathbf{H}'^{(l)} &= \text{LayerNorm}(\mathbf{H}^{(l-1)} + \mathbf{O}^{(l)}) \\
\mathbf{F}^{(l)} &= \text{FFN}(\mathbf{H}'^{(l)}) = \sigma(\mathbf{H}'^{(l)} \mathbf{W}_1^\top) \mathbf{W}_2^\top \\
\mathbf{H}^{(l)} &= \text{LayerNorm}(\mathbf{H}'^{(l)} + \mathbf{F}^{(l)})
\end{aligned}
$$

重复上述过程 $L=32$ 层，最终得到：
$$
\mathbf{H}^{(L)} \in \mathbb{R}^{6 \times d}
$$

### 2.3 生成第一个 Token

取最后一个位置的隐藏状态：
$$
\mathbf{h}_{last} = \mathbf{H}^{(L)}_{6,:} \in \mathbb{R}^{d}
$$

计算 logits：
$$
\mathbf{z} = \mathbf{h}_{last} \mathbf{W}_{lm}^\top \in \mathbb{R}^{V}
$$

其中 $\mathbf{W}_{lm} \in \mathbb{R}^{V \times d}$ 是语言模型头。

应用 softmax 得到概率分布：
$$
p(y | \mathbf{x}) = \text{softmax}(\mathbf{z}) \in \mathbb{R}^{V}, \quad \sum_{v=1}^{V} p_v = 1
$$

采样（以贪心为例）：
$$
y_1 = \arg\max_v \, p_v
$$

假设生成：**"I"**（token ID 为 100）

---

## 三、Decode 阶段（解码阶段）

现在进入自回归生成。当前序列状态：
- 已生成 token：$[x_1, \dots, x_6, y_1] = [\text{hi}, \text{,}, \text{who}, \text{are}, \text{you}, \text{?}, \text{I}]$
- 当前长度：$n_1 = 7$
- 各层 KV Cache：块 #42 中存储了 6 个 token 的 KV，剩余空间 10

### 3.1 Decode Step 1：生成第二个 token

#### 3.1.1 单 token 嵌入

输入新 token $y_1 = \text{"I"}$：
$$
\mathbf{e}_{y_1} = \text{Embedding}(y_1) \in \mathbb{R}^{d}
$$

加上位置编码（位置 7）：
$$
\mathbf{h}_{input}^{(0)} = \mathbf{e}_{y_1} + \mathbf{p}_7 \in \mathbb{R}^{d}
$$

注意：此时输入是**单个向量**，而非矩阵。

#### 3.1.2 逐层计算（关键差异：复用 KV Cache）

**第 $l$ 层计算：**

**步骤 A：生成新 token 的 Q、K、V**
$$
\begin{aligned}
\mathbf{q}_{new}^{(l)} &= \mathbf{h}_{input}^{(l-1)} \mathbf{W}_Q^{(l)\top} \in \mathbb{R}^{d} \\
\mathbf{k}_{new}^{(l)} &= \mathbf{h}_{input}^{(l-1)} \mathbf{W}_K^{(l)\top} \in \mathbb{R}^{d} \\
\mathbf{v}_{new}^{(l)} &= \mathbf{h}_{input}^{(l-1)} \mathbf{W}_V^{(l)\top} \in \mathbb{R}^{d}
\end{aligned}
$$

reshape 为多头：
$$
\mathbf{q}_{new,j}^{(l)} \in \mathbb{R}^{d_h}, \quad \mathbf{k}_{new,j}^{(l)} \in \mathbb{R}^{d_h}, \quad \mathbf{v}_{new,j}^{(l)} \in \mathbb{R}^{d_h} \quad \text{for } j=1,\dots,h
$$

**步骤 B：追加 KV 到 Cache**

检查块 #42 是否有空间：已用 6，容量 16，有空间。

追加写入：
$$
\begin{aligned}
\text{Block}_{42}.\mathbf{K}[6,:] &= \mathbf{k}_{new}^{(l)} \\
\text{Block}_{42}.\mathbf{V}[6,:] &= \mathbf{v}_{new}^{(l)}
\end{aligned}
$$

现在块 #42 包含 7 个 KV 向量。

**步骤 C：PagedAttention 计算（核心优化）**

需要从块 #42 读取所有历史 KV（7 个向量）来计算注意力。

对每个头 $j$，构造完整的 K 和 V：
$$
\mathbf{K}_{full,j} = \begin{bmatrix} 
\mathbf{k}_{1,j}^{(l)\top} \\ 
\vdots \\ 
\mathbf{k}_{7,j}^{(l)\top} 
\end{bmatrix} \in \mathbb{R}^{7 \times d_h}, \quad
\mathbf{V}_{full,j} = \begin{bmatrix} 
\mathbf{v}_{1,j}^{(l)\top} \\ 
\vdots \\ 
\mathbf{v}_{7,j}^{(l)\top} 
\end{bmatrix} \in \mathbb{R}^{7 \times d_h}
$$

注意力分数（单查询对多 key）：
$$
\mathbf{s}_j = \frac{\mathbf{q}_{new,j}^{(l)} \mathbf{K}_{full,j}^\top}{\sqrt{d_h}} \in \mathbb{R}^{7}
$$

Softmax：
$$
\mathbf{a}_j = \text{softmax}(\mathbf{s}_j) \in \mathbb{R}^{7}, \quad \sum_{i=1}^{7} a_{j,i} = 1
$$

注意力输出：
$$
\mathbf{o}_{new,j}^{(l)} = \mathbf{a}_j \mathbf{V}_{full,j} = \sum_{i=1}^{7} a_{j,i} \cdot \mathbf{v}_{i,j}^{(l)} \in \mathbb{R}^{d_h}
$$

拼接所有头并投影：
$$
\mathbf{o}_{new}^{(l)} = \text{Concat}[\mathbf{o}_{new,1}^{(l)}, \dots, \mathbf{o}_{new,h}^{(l)}] \mathbf{W}_O^{(l)\top} \in \mathbb{R}^{d}
$$

**步骤 D：残差与 FFN**
$$
\begin{aligned}
\mathbf{h}'^{(l)}_{input} &= \text{LayerNorm}(\mathbf{h}_{input}^{(l-1)} + \mathbf{o}_{new}^{(l)}) \\
\mathbf{f}^{(l)} &= \text{FFN}(\mathbf{h}'^{(l)}_{input}) \\
\mathbf{h}_{input}^{(l)} &= \text{LayerNorm}(\mathbf{h}'^{(l)}_{input} + \mathbf{f}^{(l)})
\end{aligned}
$$

经过 $L$ 层后得到：
$$
\mathbf{h}_{output} = \mathbf{h}_{input}^{(L)} \in \mathbb{R}^{d}
$$

#### 3.1.3 生成第二个 token

$$
\mathbf{z} = \mathbf{h}_{output} \mathbf{W}_{lm}^\top \in \mathbb{R}^{V}
$$

$$
y_2 = \arg\max_v \, \text{softmax}(\mathbf{z})_v
$$

假设生成：**"am"**

---

### 3.2 Decode Step 2：生成第三个 token

当前序列长度 $n_2 = 8$。

输入：$y_2 = \text{"am"}$

**关键操作：**

1. **嵌入**：$\mathbf{e}_{y_2} + \mathbf{p}_8$

2. **KV Cache 追加**：块 #42 已用 7，追加后变为 8

3. **PagedAttention**：从块 #42 读取 8 个 KV 向量
   $$
   \mathbf{K}_{full} \in \mathbb{R}^{8 \times d}, \quad \mathbf{V}_{full} \in \mathbb{R}^{8 \times d}
   $$

4. **注意力计算**：
   $$
   \mathbf{s} = \frac{\mathbf{q}_{new} \mathbf{K}_{full}^\top}{\sqrt{d_h}} \in \mathbb{R}^{8}
   $$
   注意：查询始终是单个向量，key 序列随长度增长。

生成 $y_3 = \text{"an"}$（假设）

---

### 3.3 Decode Step 3：生成第四个 token

当前长度 $n_3 = 9$。

生成 $y_4 = \text{"AI"}$（假设）

---

### 3.4 Decode Step 4：生成第五个 token

当前长度 $n_4 = 10$。

生成 $y_5 = \text{"assistant"}$（假设）

---

### 3.5 Decode Step 5：遇到 EOS 停止

生成 $y_6 = \text{""}$（EOS token）

停止生成。

---

## 四、完整序列的数学表示

最终生成的完整序列：
$$
\mathbf{y} = [y_1, y_2, y_3, y_4, y_5, y_6] = [\text{I}, \text{am}, \text{an}, \text{AI}, \text{assistant}, \text{}]
$$

完整对话：
$$
\text{Input: "hi, who are you"} \rightarrow \text{Output: "I am an AI assistant"}
$$

---

## 五、关键公式对比总结

| 阶段 | Prefill | Decode Step $t$ |
|------|---------|-----------------|
| **输入维度** | $\mathbf{X} \in \mathbb{R}^{n \times d}$ | $\mathbf{x}_{new} \in \mathbb{R}^{d}$（单向量）|
| **Q 维度** | $\mathbf{Q} \in \mathbb{R}^{n \times d}$ | $\mathbf{q}_{new} \in \mathbb{R}^{d}$ |
| **K/V 计算** | 全序列计算 $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{n \times d}$ | 仅计算新 token：$\mathbf{k}_{new}, \mathbf{v}_{new} \in \mathbb{R}^{d}$ |
| **KV Cache 操作** | 创建并写入块 | 读取历史 + 追加新 KV |
| **注意力分数** | $\mathbf{S} \in \mathbb{R}^{n \times n}$（矩阵） | $\mathbf{s} \in \mathbb{R}^{n+t}$（向量）|
| **计算复杂度** | $O(n^2 \cdot d)$ | $O((n+t) \cdot d)$ per step |
| **内存访问** | 写入 $n$ 个 KV | 读取 $(n+t)$ 个 KV + 写入 1 个 KV |

---

## 六、PagedAttention 的内存布局可视化

```
物理内存块 #42（大小 B=16）:
┌─────────────────────────────────────────────────────────────┐
│  位置:  0    1    2    3    4    5    6    7    8    9  ... 15 │
├─────────────────────────────────────────────────────────────┤
│ Prefill: [hi] [,] [who][are][you][ ? ]                      │
│          │    │    │    │    │    │                         │
│          K1   K2   K3   K4   K5   K6   ← 存储的 Key 向量      │
│          V1   V2   V3   V4   V5   V6   ← 存储的 Value 向量    │
├─────────────────────────────────────────────────────────────┤
│ Decode1:                      [ I ]                         │
│                                │                            │
│                               K7   ← 追加                     │
│                               V7   ← 追加                     │
├─────────────────────────────────────────────────────────────┤
│ Decode2:                           [am]                     │
│                                     │                       │
│                                    K8  ← 追加                 │
│                                    V8  ← 追加                 │
├─────────────────────────────────────────────────────────────┤
│ ... 继续直到块满或生成结束                                       │
└─────────────────────────────────────────────────────────────┘

块表 Block Table: [42]  （逻辑块 0 → 物理块 42）
```

当序列超过 16 个 token 时，系统会分配第二个物理块（如 #57），块表变为 $[42, 57]$，注意力内核通过块表非连续读取 KV。

---

## 七、核心优化点总结

1. **Prefill 的并行性**：利用矩阵乘法一次性计算 $n \times n$ 注意力矩阵，充分发挥 GPU 算力。

2. **Decode 的 KV 复用**：避免重复计算历史 token 的 K、V，将 $O(n^2)$ 降为 $O(n)$ 每步。

3. **分页内存管理**：通过块表实现非连续存储，支持动态长度、内存共享和高效批处理。

4. **内存带宽优化**：PagedAttention 内核通过分块加载和共享内存，减少全局内存访问次数，缓解 Decode 阶段的带宽瓶颈。