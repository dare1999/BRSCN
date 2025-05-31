# dare
# BRSCN: Behavior Representation Spatial Correlation Network

本仓库实现了论文 **“基于行为表征空间关联网络的综合交通枢纽客流短时预测框架（BRSCN）”** 中提出的客流预测模型。BRSCN 结合虚拟现实（VR）实验采集的旅客活动链、图注意力网络（GAT）、双向扩展 LSTM（Bi-sLSTM）和动态窗口稀疏注意力 Transformer，有效捕捉枢纽内非邻接区域间的复杂时空交互，从而实现短时客流量的高精度预测。

> **引用**：匿名版_基于行为表征空间关联网络的综合交通枢纽客流短时预测 

![image](https://github.com/user-attachments/assets/10b4b348-c96a-4ec3-ac8c-682c1064f3d1)

VR Platform experimental framework based on real comprehensive transportation hub scenario data

![image](https://github.com/user-attachments/assets/29b5637e-a31a-4617-94eb-8911e1bac236)

Overall framework of hub passenger flow distribution prediction network BRSCN

---

## 目录

- [1. 模型概述](#1-模型概述)  
- [2. 关键模块](#2-关键模块)  
  - [2.1 空间关联图构建](#21-空间关联图构建)  
  - [2.2 图注意力网络（GAT）](#22-图注意力网络gat)  
  - [2.3 双向扩展 LSTM（Bi-sLSTM）](#23-双向扩展-lstmbislstm)  
  - [2.4 动态窗口稀疏注意力 Transformer](#24-动态窗口稀疏注意力-transformer)  
  - [2.5 模型融合与输出](#25-模型融合与输出)  
- [3. 主要超参数与实现细节](#3-主要超参数与实现细节)  
  - [3.1 空间关联图参数](#31-空间关联图参数)  
  - [3.2 GAT 模块参数](#32-gat-模块参数)  
  - [3.3 Bi-sLSTM 模块参数](#33-bi-slstm-模块参数)  
  - [3.4 动态窗口 Transformer 参数](#34-动态窗口-transformer-参数)  
  - [3.5 训练超参数](#35-训练超参数)  
- [4. 数据预处理](#4-数据预处理)  
- [5. 使用示例](#5-使用示例)  
- [6. 仓库结构（示例）](#6-仓库结构示例)  
- [7. 参考文献](#7-参考文献)  

---

## 1. 模型概述

BRSCN 的设计目标是针对综合交通枢纽内部多类型功能区域客流预测中，非邻接区域之间时空关联难以捕捉的问题，提出一种结合旅客行为链重构与图神经网络的短时客流预测框架。整体流程如下：

1. **旅客行为链重构 & 图嵌入**：  
   - 利用 VR 环境采集枢纽内部旅客的三维空间轨迹，重构活动链（Mobility Chain），并根据相邻时刻入/离区域判定函数生成功能区节点序列。  
   - 通过 Skip-gram 模型对所有恢复的活动链进行滑动窗口采样，学习每个功能区的“行为表征”向量；再结合区域间旅客转移频率，构建有权无向的空间关联图 $\mathcal{G}=(V,E)$ 。  
2. **图注意力网络（GAT）**：  
   - 输入空间关联图邻接矩阵与节点初始特征（历史客流量或行为嵌入），通过多头 GAT 聚合邻接节点（含非邻接语义关联区域）的特征，生成空间编码。  
   - 紧接时序卷积（TCN）模块，对 GAT 输出按功能区维度做一维卷积，提炼局部空间图特征。  
3. **双向扩展 LSTM（Bi-sLSTM）**：  
   - 将 TCN 后的空间特征与历史客流时序特征在时间维度拼接形成输入序列，送入 Bi-sLSTM。  
   - sLSTM 在传统 LSTM 基础上引入“指数门”（Exponential Gate），对输入门、遗忘门和输出门采用指数激活，以提升非线性表达能力与梯度传播效率。  
   - Bi-sLSTM 对序列进行前向与后向编码，最后将两个方向的隐藏状态按元素加和，得到综合时序编码向量。  
4. **动态窗口稀疏注意力 Transformer**：  
   - 以 Bi-sLSTM 输出作为输入，将时刻 $t$ 的特征映射到 Query/Key/Value 空间。  
   - 使用指数加权移动平均（EWMA）计算步长 $t$ 的客流“波动幅度” $\lambda_t$，并通过 Sigmoid 映射得到局部窗口大小 $\omega_t$（$\omega_\text{min}=24,\;\omega_\text{max}=120$）。  
   - 生成稀疏注意力掩码，将非窗口覆盖位置置为 $-∞$，只在局部窗口范围内执行点积注意力。多头注意力输出再通过线性映射得到最终时序特征。  
5. **输出层（全连接）**：  
   - 将 Transformer 输出经残差连接、LayerNorm 处理后，送入多层全连接网络（MLP），映射到未来短时 $L'$ 个时间步各功能区的客流量预测值。

该框架在上海虹桥综合交通枢纽高架层真实数据集上验证，相较于 AGCRN、STTN、ST-MAN 等多种基线模型，在 RMSE、MAE、MAPE 上分别降低约 30.2%、21.1% 和 28.3%；同时引入空间关联图后 RMSE 再下降 16.7% 。

---

## 2. 关键模块

### 2.1 空间关联图构建

- **数据来源**：  
  - VR 实验中采集的旅客在综合枢纽场景中的 1s 采样三维轨迹。  
  - 实验招募 60 名受试者，共采集 360 条轨迹，覆盖枢纽内 $K=20$ 个功能区。  

- **区域识别与链路重构**：  
  - 设第 $i$ 名旅客在时刻 $t'$ 的空间坐标为 $\mathbf{p}_i(t')=(x_i(t'),y_i(t'),z_i(t'))$；设各功能区边界为 $\{R_k\}_{k=1}^K$。  
  - 通过判定函数  
    $$
      \delta_i^k(t') =
      \begin{cases}
        1, & \mathbf{p}_i(t')\in R_k,\\
        0, & \text{otherwise},
      \end{cases}
    $$  
    判断旅客进入或离开某区域。若停留时长 $\Delta t_{i,k}\ge\theta_k=6\mathrm{s}$ 且满足问卷标注的服务需求条件，则判定为在区域 $k$ 停留并接受服务。  
  - 按首次进入时间顺序将各旅客在站内经过的功能区编码连成一条“游走链路”序列 $\mathbf{s}_n=(v_{n,1},v_{n,2},\dots,v_{n,T_n})$，$v_{n,m}\in V$。  

- **节点嵌入（Skip-gram）**：  
  - 对所有 $N$ 条活动链作滑动窗口采样，窗口大小 $\omega$。对每个中心节点 $v_{i,m}$，采样其上下文节点 $\mathcal{N}(v_{i,m}) = \{\,v_{i,c}:\,|c-m|\le \omega\}\,$。  
  - 最小化负对数似然：  
    $$
      \min -\sum_{i=1}^N \sum_{c\in \mathcal{N}(v_{i,m})} \log p(v_{i,c}\mid v_{i,m}),
      \quad
      p(v_{i,c}\mid v_{i,m}) = \frac{\exp(\mathbf{u}_{v_{i,m}}\cdot \mathbf{u}_{v_{i,c}})}
      {\sum_{v\in V}\exp(\mathbf{u}_{v_{i,m}}\cdot \mathbf{u}_v)}.
    $$  
  - 训练结束后得到每个功能区节点 $v\in V$ 的初始行为嵌入向量 $\mathbf{u}_v\in \mathbb{R}^{d_u}$。  

- **边权计算**：  
  - 统计游走链中从区域 $a$ 到 $b$ 的总转移次数 $\eta_{a,b}$，计算转移频率 $\varpi_{a,b}=\sum \eta_{a,b}/\sum_{(i,j)\in V\times V}\eta_{i,j}$。  
  - 计算节点嵌入向量余弦相似度 $\cos(\mathbf{u}_a,\mathbf{u}_b)$.  
  - 根据公式  
    $$
      w_{a,b} =
      \begin{cases}
        \beta\,\cos(\mathbf{u}_a,\mathbf{u}_b) + (1-\beta)\,\varpi_{a,b}, & \cos(\mathbf{u}_a,\mathbf{u}_b)\ge 0,\\
        0, & \text{otherwise},
      \end{cases}
    $$  
    生成有权无向边 $\,e_{a,b}\,$，其中超参数 $\beta=0.6$（以最小 RMSE 原则选取）:contentReference[oaicite:3]{index=3}。  
  - 最终得到区域空间关联图 $\mathcal{G}=(V, E, W)$，其中 $|V|=K$，邻接矩阵 $A\in\mathbb{R}^{K\times K}$，$A_{a,b}=w_{a,b}$。  

> **提示**：在示例代码 `load_graph_data('graph.csv')` 中，可读取预先计算好的邻接矩阵；若维度不符，则按客流列相关性阈值 $\rho>0.5$ 重构二值邻接矩阵。:contentReference[oaicite:4]{index=4}

---

### 2.2 图注意力网络（GAT）

对空间关联图节点特征进行聚合，生成区域空间编码：

1. **输入**  
   - 节点特征初始值：  
     - 如果使用漫游链行为嵌入：$\mathbf{x}_a^{(0)} = \mathbf{u}_a\in\mathbb{R}^{16}$。  
     - 如果使用历史客流量：$\mathbf{x}_a^{(0)} = [f_a(t-L+1),\,\dots,\,f_a(t)]\in\mathbb{R}^{L}$。  
   - 邻接矩阵 $A$ 或边索引 `edge_index`（代码为 PyTorch Geometric 张量格式）。  

2. **多头注意力机制**  
   - 对第 $\ell$ 层，第 $h$ 个注意力头，节点 $a$ 与邻居 $b\in\mathcal{N}(a)$ 计算未归一化注意力分数：  
     $$
       e_{a,b}^{(h)} = \text{LeakyReLU}\bigl(\,[\mathbf{W}\,\mathbf{x}_a^{(\ell)} \,\|\, \mathbf{W}\,\mathbf{x}_b^{(\ell)}]\cdot \boldsymbol{\delta}_h \bigr),
     $$  
     其中 $\mathbf{W}\in\mathbb{R}^{d_g\times d_\text{in}}$ 为线性投影，$\boldsymbol{\delta}_h\in\mathbb{R}^{2d_g}$ 为注意力向量，$\|\,$ 表示拼接操作。:contentReference[oaicite:5]{index=5}  
   - 归一化：  
     $$
       \alpha_{a,b}^{(h)} = \frac{\exp\bigl(e_{a,b}^{(h)}\bigr)}{\sum_{b'\in\mathcal{N}(a)} \exp\bigl(e_{a,b'}^{(h)}\bigr)}.
     $$  
   - 每个注意力头输出：  
     $$
       \mathbf{z}_a^{(h)} = \sum_{b\in \mathcal{N}(a)} \alpha_{a,b}^{(h)}\,\mathbf{W}\,\mathbf{x}_b^{(\ell)}.
     $$  
   - 多头拼接：$\mathbf{z}_a = \bigl\|_{h=1}^H\,\mathbf{z}_a^{(h)}\in\mathbb{R}^{H\,d_g}$.  

3. **时序卷积（TCN）**  
   - 将多头 GAT 输出 $\{\mathbf{z}_a\}_{a=1}^K$ 重组为形状 $[\,K\times d_\text{out}\,]$，再 reshape 为 $[\,1,\,d_\text{out},\,K\,]$（每个通道对应一个区域序列）。  
   - 通过一维因果卷积核大小 $g_k$（代码中 `g_k=3`）进行卷积，并做 LayerNorm + Dropout + Flatten，得到每个节点 $\tilde{\mathbf{z}}_a\in\mathbb{R}^{d_\text{spat}}$。  
   - 最终 GAT 输出 $\mathbf{H}_\text{spat} = [\,\tilde{\mathbf{z}}_1,\,\dots,\,\tilde{\mathbf{z}}_K\,]\in\mathbb{R}^{K\times d_\text{spat}}$。

---

### 2.3 双向扩展 LSTM（Bi-sLSTM）

对空间编码与历史时序特征进行序列编码：

1. **输入拼接**  
   - 对于时刻 $t$，将每个区域 $a$ 的空间编码 $\tilde{\mathbf{z}}_a(t)$ 与其历史客流时序 $\mathbf{f}_a(t-L+1:t)\in\mathbb{R}^{L}$ 在特征维度拼接，形成 $[\mathbf{z}_a\;\|\;\mathbf{f}_a]\in\mathbb{R}^{L + d_\text{spat}}$。  
   - 对所有 $K$ 个区域同时构成形状 `[batch_size, seq_len=L, num_nodes=K, feature_dim=L + d_spat]` 的张量。  

2. **sLSTM 单元**  
   - 在每个时间步，输入为 $\mathbf{f}_t\in\mathbb{R}^{d_\text{in}}$，隐藏状态为 $\mathbf{h}_{t-1}$、细胞状态 $\mathbf{c}_{t-1}$。  
   - 通过指数门（Exponential Gate）增强非线性：  
     ```math
       \tilde{\mathbf{z}}_t = \tanh\bigl(\mathbf{W}_z \mathbf{f}_t + \mathbf{R}_z \mathbf{h}_{t-1} + b_z\bigr), \\
       \hat{\mathbf{i}}_t = \exp\bigl(\mathbf{W}_i \mathbf{f}_t + \mathbf{R}_i \mathbf{h}_{t-1} + b_i\bigr), \\
       \hat{\mathbf{f}}_t = \sigma\bigl(\mathbf{W}_f \mathbf{f}_t + \mathbf{R}_f \mathbf{h}_{t-1} + b_f\bigr), \\
       \hat{\mathbf{o}}_t = \sigma\bigl(\mathbf{W}_o \mathbf{f}_t + \mathbf{R}_o \mathbf{h}_{t-1} + b_o\bigr), \\
       \mathbf{c}_t = \hat{\mathbf{f}}_t \odot \mathbf{c}_{t-1} + \hat{\mathbf{i}}_t \odot \tilde{\mathbf{z}}_t, \\
       \mathbf{h}_t = \hat{\mathbf{o}}_t \odot \tanh(\mathbf{c}_t),
     ```  
     其中 $\hat{\mathbf{i}}_t,\hat{\mathbf{f}}_t,\hat{\mathbf{o}}_t$ 分别为指数激活或 Sigmoid 激活结果，相较传统 LSTM 引入更强非线性提升序列建模能力。  

3. **双向编码**  
   - 前向与后向两个 sLSTM 并行处理同一序列，得到正向隐藏 $\mathbf{h}_t^{(f)}$ 与反向隐藏 $\mathbf{h}_t^{(b)}$。  
   - 将两个方向的隐藏状态逐元素相加：$\mathbf{h}_t^\text{bi} = \mathbf{h}_t^{(f)} + \mathbf{h}_t^{(b)}$，作为时刻 $t$ 的最终时序编码。  

4. **输出**  
   - 取最后一个时间步 $t=L$ 的双向隐藏 $\mathbf{h}_L^\text{bi}\in\mathbb{R}^{d_\text{hid}}$ 作为该序列的 Bi-sLSTM 输出。  

---

### 2.4 动态窗口稀疏注意力 Transformer

对 Bi-sLSTM 输出的序列特征进行多头局部注意力编码，降低计算复杂度且自适应波动：

1. **输入投影**  
   - 将 Bi-sLSTM 在所有时间步的隐藏 $\bigl\{\mathbf{h}_t^\text{bi}\bigr\}_{t=1}^L$ 堆叠为张量 $\mathbf{X}\in\mathbb{R}^{L\times d_\text{hid}}$。  
   - 线性投影得到 Queries/Keys/Values：  
     $$
       \mathbf{Q} = \mathbf{X}\,\mathbf{W}_q,\quad
       \mathbf{K} = \mathbf{X}\,\mathbf{W}_k,\quad
       \mathbf{V} = \mathbf{X}\,\mathbf{W}_v,\quad
       \mathbf{W}_q,\mathbf{W}_k,\mathbf{W}_v\in\mathbb{R}^{d_\text{hid}\times d_\text{hid}}.
     $$  

2. **EWMA 波动计算**  
   - 计算当前时刻 $t$ 的波动幅度：$\lambda_t = \|\mathbf{h}_t^\text{bi} - \mathbf{h}_{t-1}^\text{bi}\|_1$（仅当 $t>1$ 时）。  
   - 计算全局平均波动：$\bar{\lambda} = \frac{1}{L}\sum_{t=2}^L \lambda_t$.  
   - 使用指数加权移动平均：  
     $$
       \tilde{\lambda}_t = \gamma\,\lambda_t + (1-\gamma)\,\bar{\lambda},\quad
       \gamma=0.525\text{（可学习，实验后收敛值）}.  
     $$  
     :contentReference[oaicite:8]{index=8}  

3. **动态窗口大小**  
   - 将 $\tilde{\lambda}_t$ 通过 Sigmoid 映射至 $(0,1)$，计算自适应窗口：  
     $$
       \omega_t = \omega_\text{min} + (\omega_\text{max} - \omega_\text{min}) \cdot \sigma(\tilde{\lambda}_t),  
       \quad \omega_\text{min}=24,\;\omega_\text{max}=120.  
     $$  
   - 将 $\omega_t$ 向下取整为整数，并保证其在 $[\omega_\text{min},\,\omega_\text{max}]$ 范围内。  

4. **生成稀疏掩码**  
   - 针对长度 $L$，构造掩码矩阵 $M\in\mathbb{R}^{L\times L}$：  
     $$
       M_{i,j} =
       \begin{cases}
         0, & \text{if } |i-j|\le \lfloor \omega_t/2\rfloor,\\
         -\infty, & \text{otherwise}.
       \end{cases}
     $$  
   - 此处仅保留局部窗口范围内的注意力交互。  

5. **多头稀疏注意力**  
   - 将 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 重塑为 `[batch_size=1, heads=H, seq_len=L, head_dim=d_h]`。  
   - 计算缩放点积：  
     $$
       \mathbf{S} = \frac{\mathbf{Q}\,\mathbf{K}^\top}{\sqrt{d_h}} + M,\quad
       \mathbf{A} = \text{softmax}(\mathbf{S}),\quad
       \mathbf{O}^{(h)} = \mathbf{A}^{(h)}\,\mathbf{V}^{(h)},\;\forall\,h\in\{1,\dots,H\}.
     $$  
   - 每个头的输出 $\mathbf{O}^{(h)}\in \mathbb{R}^{L\times d_h}$，多头拼接后通过可训练矩阵 $\mathbf{W}_o$ 加权融合：  
     $$
       \mathbf{O}_\text{multi} = \bigl[\mathbf{O}^{(1)},\dots,\mathbf{O}^{(H)}\bigr]\;\times\;\mathbf{W}_o,\quad
       \mathbf{W}_o\in\mathbb{R}^{H\,d_h\times d_\text{hid}}.
     $$  

6. **Add & Norm + Feed-Forward**  
   - 残差连接：$\mathbf{X}' = \text{LayerNorm}\bigl(\mathbf{X} + \mathbf{O}_\text{multi}\bigr)$.  
   - 前馈网络（两层线性 + ReLU + Dropout）：  
     $$
       \mathbf{X}'' = \text{LayerNorm}\bigl(\mathbf{X}' + \text{FFN}(\mathbf{X}')\bigr),  
     $$  
     其中 $\text{FFN}(x) = W_2\,\text{ReLU}(W_1\,x + b_1) + b_2,\;W_1\in\mathbb{R}^{d_\text{hid}\times d_{ff}},\,W_2\in\mathbb{R}^{d_{ff}\times d_\text{hid}}.  

7. **输出**  
   - 取最后一个时间步 $t=L$，即 $\mathbf{X}''_{L}\in\mathbb{R}^{d_\text{hid}}$，作为 Transformer 编码后的时序特征。  

> **说明**：代码实现见 `DynamicWindowSparseAttention` 和 `TransformerEncoderLayerWithResidual` 类，其中默认 `max_window=120, min_window=24, heads=4, \gamma` 可学习参数初始值 0.1，最终训练收敛为 0.525。

---

### 2.5 模型融合与输出

1. **STGAT + Temporal Attention**  
   - 由于 STGAT 内部已融合 GAT + TCN，代码中采用 `STGAT` 类先对每个时间步的节点特征进行空间图聚合，得到序列 $\{\mathbf{Z}_t\}_{t=1}^L$。  
   - 之后通过 `TemporalAttention` 类，对整个序列做多头时序注意力（传统稠密注意力），进一步增强时序内各时刻间依赖表示。  
   - 其输出与行为嵌入 $\mathbf{u}_v$ 以及时间特征（如“星期几”、“小时”等经线性映射得到的嵌入）在特征维度拼接，形成特征融合向量。

2. **xLSTMWithTransformer**  
   - 将上述融合特征先做线性映射（`feature_fusion`），再依次通过 `xLSTMBlockWithResidual`（sLSTM + 残差 + LayerNorm + Dropout）和 `TransformerEncoderLayerWithResidual`（动态窗口稀疏注意力 + 残差 + LayerNorm + FFN）模块迭代编码。  
   - 取最后时间步的隐藏输出 $\mathbf{h}_L^\text{trans}\in\mathbb{R}^{d_\text{hid}}$，送入多层全连接（`output_layer`），预测下一时刻或多时刻的客流值（在代码中只预估下一时刻）。  

3. **多功能区并行预测**  
   - 由于 `xLSTMWithTransformer.forward` 中 `input_seq` 形状为 `[batch_size, seq_len, num_nodes, 1]`，并行对每个节点（区域）进行处理，最终 `final_output` 形状为 `[batch_size, num_nodes]`，即可一次性输出 $K$ 个区域在下一时刻的客流预测（或多个时刻，需自行扩展输出维度）。  

---

## 3. 主要超参数与实现细节

以下均来自论文与示例代码中“最佳”复现配置，供复现与二次开发时参考。

### 3.1 空间关联图参数

- **行为嵌入维度**：$d_u = 16$（Skip-gram 输出向量维度）。  
- **活动链滑动窗口宽度**：$\omega = 5$（实验中常用值，可根据轨迹长度和节点数调整）。  
- **空间关联权重超参数**：$\beta = 0.6$（以 RMSE 最小化准则选取）。  
- **图中节点数**：$K = 20$（上海虹桥枢纽高架层共 20 个功能区域）。  

> :contentReference[oaicite:10]{index=10}

### 3.2 GAT 模块参数

- **输入通道数**：若使用行为嵌入，则 `in_channels = 16`；若使用真实客流序列，则 `in_channels = L=7`（示例中先前七步）。  
- **输出通道数**：`out_channels = 16`。  
- **多头数（第 1 层）**：`heads = 2`，将输出维度拆分为每头 $8$ 维，再拼接回 $16$ 维。  
- **层数**：`layers = 1`（示例中仅一层 GAT + 紧接一层单头 GAT 做融合）。  
- **激活**：第一层 `GATConv` 后使用 `ELU`，第二层为线性变换（`concat=False`）。  
- **Dropout**：`p=0.3`。  
- **TCN 卷积核尺寸**：`temporal_kernel = 3`，`padding="same"` 保持维度不变。  
- **LayerNorm 维度**：`d_out = 16`。  
- **残差**：若 `in_channels != out_channels`，则通过 `Linear(in_channels, out_channels)` 实现维度对齐。  

> 

### 3.3 Bi-sLSTM 模块参数

- **Embedding 输入维度**：在 _xLSTMWithTransformer_ 中，`embedding_size = 32`（线性融合后输出维度）。  
- **隐藏单元数**：`hidden_size = 32`（示例代码）。但在论文实验（20 个区域场景）中，设置 `hidden_size = 80` 以获得最佳 RMSE/MAE/MAPE :contentReference[oaicite:12]{index=12}。  
- **LSTM 层数**：`num_layers = 1`。  
- **双向**：`bidirectional = True`。  
- **Dropout**：`dropout = 0.1`。  
- **sLSTM 指数门初始化**：指数门参数初始化为 0.1，训练中优化最终收敛为 $\gamma \approx 0.525$。  

> 

### 3.4 动态窗口 Transformer 参数

- **模型维度**：`d_model = hidden_size = 32`。  
- **多头数**：`num_heads = 4`（每头维度 $d_h = 32/4=8$）。  
- **FFN 隐藏层维度**：`dim_feedforward = 2048`（可根据显存与收敛速度调整）。  
- **头部权重矩阵**：`W_o` 维度 `[num_heads × head_dim × head_dim] = [4 × 8 × 8]`，用于多头输出加权。  
- **最大/最小窗口**：`max_window = 120, min_window = 24`。  
- **Dropout**：`dropout = 0.1`。  
- **LayerNorm ϵ**：`layer_norm_eps = 1e-5`。  

> 

### 3.5 训练超参数

- **历史序列长度**：`L = 7`（代码中通过 `window_size=7` 生成数据窗口）。  
- **批大小（BatchSize）**：`batch_size = 512`（论文设置）或 `batch_size = 16`（示例代码训练时可减小以避免显存不足）。  
- **学习率（LR）**：`lr = 1×10^{-3}`（论文设置）或示例代码中 `lr = 1×10^{-4}`。  
- **优化器**：`Adam`（论文）或 `AdamW`（示例代码，`weight_decay=1×10^{-2}`）。  
- **学习率调度**：论文未显式提及 Scheduler；示例代码使用 `CosineAnnealingLR`，`T_max=100`。  
- **训练轮数（Epoch）**：`num_epochs = 300`。  
- **梯度裁剪**：若梁度梯度爆炸，可采用 `max_norm=1.0`。  
- **早停机制**：验证集 `patience = 20`。  

> 

---

## 4. 数据预处理

1. **原始数据**：CSV 文件包含 20 个功能区在采样时段内（如 2024-08-22 08:00–20:00）每 10s 一次的人流量统计，共 $N$ 条时间戳。  
2. **特征归一化**：  
   - 使用 `MinMaxScaler` 分别对节点（各区域当时人流量）、时间特征（weekday, hour 映射后得到 `day_of_week_sin`, `day_of_week_cos`, `hour_sin`, `hour_cos`）以及标签（$\log1p$ 后再归一化客流）做归一化。  
3. **时间特征工程**：  
   - 若原始数据无时间戳列，则可通过逻辑添加滞后特征（如 `F_i_lag1`、`F_i_lag2`、`F_i_lag3`、`F_i_lag7`）并随机填充日期/小时。  
4. **时间窗口划分**：  
   - 使用 `create_windows(x, y, temporal, window_size=7)` 生成滑动窗口序列，构成 `x_windows.shape=(N-7, 7, 20, 1)`，`y_windows.shape=(N-7, 1)`，`t_windows.shape=(N-7,7,4)`。  
5. **图数据导入**：  
   - 从 `graph.csv` 中读取 $20\times 20$ 邻接矩阵。若维度与预期不符，可按客流列相关性阈值 $\rho>0.5$ 重建。  
   - 生成 `edge_index`：遍历矩阵索引 $(i,j)$，若 $A_{i,j}>0$ 或 $i=j$，则将 $(i,j)$ 加入边索引列表；返回形状为 `[2,\,E]` 的 LongTensor。  


---

