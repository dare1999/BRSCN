import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch_geometric.nn as gnn
from torch_geometric.data import Data
import math
import torch.nn.functional as F

# 设置随机数种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

############################################
# 图注意力网络模块
############################################
class STGAT(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_kernel=3, heads=4):
        super().__init__()
        # 空间图注意力层
        self.gat1 = gnn.GATConv(in_channels, out_channels // heads, heads=heads, dropout=0.1)
        self.gat2 = gnn.GATConv(out_channels, out_channels, heads=1, concat=False, dropout=0.1)
        # 时间卷积层
        self.tcn = nn.Conv1d(out_channels, out_channels, kernel_size=temporal_kernel, padding="same")
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        # 空间聚合
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.gat2(x, edge_index)  # [batch*num_nodes, out_channels]

        # 时间卷积
        batch_size = x.shape[0] // 4  # 假设固定4个节点
        x = x.view(batch_size, 4, -1).permute(0, 2, 1)  # [batch, C, num_nodes]
        x = self.tcn(x)  # [batch, C, num_nodes]
        x = x.permute(0, 2, 1).reshape(-1, x.shape[1])  # [batch*num_nodes, C]
        return self.norm(x)

############################################
# 动态窗口稀疏注意力机制及Transformer和LSTM模块
############################################
class DynamicWindowSparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_window=120, min_window=24, alpha=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_window = max_window
        self.min_window = min_window
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.head_dim = d_model // num_heads

        # 可训练参数
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.W_o = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim))  # 多头权重矩阵

        # 初始化EWMA状态
        self.register_buffer('ewma', torch.zeros(1))
        self.register_buffer('step', torch.tensor(0))
        self.register_buffer('ewma_prev', torch.zeros(1))  # 保存上一步的EWMA值
        self.register_buffer('step', torch.tensor(0))

    def _calc_fluctuation(self, x):
        """
        同时返回：
        - curr_fluct: 当前时刻与前一时刻每个样本的波动值（张量）
        - curr_fluct_mean: 所有样本的平均波动（标量），用于更新 ewma
        """
        curr_fluct = torch.abs(x[:, -1] - x[:, -2])  # [batch_size, d_model]
        curr_fluct_mean = curr_fluct.mean()  # scalar
        return curr_fluct, curr_fluct_mean

    def _update_ewma(self, curr_fluct):
        """
        更新EWMA：用当前step对应的波动值与历史EWMA做加权
        """
        # 防止step超过batch size，使用 clamp 限制索引
        alpha = torch.sigmoid(self.alpha)
        print(alpha)
        fluct_step_value = curr_fluct[0, self.step % curr_fluct.size(1)] if curr_fluct.dim() == 2 else curr_fluct[
            self.step % curr_fluct.size(0)]

        if self.step == 0:
            self.ewma_prev = fluct_step_value.clone().detach()
        else:
            self.ewma_prev = alpha * fluct_step_value + (1 - alpha) * curr_fluct.mean()
        self.step += 1

    def _get_window_size(self, curr_fluctuation):
        """
        将历史与当前波动加权混合，生成窗口大小（使用均值保证为标量）
        """
        alpha = torch.sigmoid(self.alpha)
        if curr_fluctuation.numel() > 1:
            curr_mean = curr_fluctuation.mean()
        else:
            curr_mean = curr_fluctuation

        window_signal = alpha * curr_mean + (1 - alpha) * self.ewma_prev.mean()
        scale_factor = torch.sigmoid(window_signal)
        window_size = self.min_window + (self.max_window - self.min_window) * scale_factor
        return int(window_size.item())

    def _generate_mask(self, seq_len, window_size, device):
        """生成动态窗口掩码"""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 0
        return mask

    def forward(self, x):
        # 计算客流波动并更新EWMA
        curr_fluctuation, _ = self._calc_fluctuation(x)
        self._update_ewma(curr_fluctuation)
        window_size = self._get_window_size(curr_fluctuation)

        # 投影到查询、键、值空间
        batch_size, seq_len, _ = x.size()
        queries = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 计算注意力得分
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用动态窗口掩码
        mask = self._generate_mask(seq_len, window_size, x.device)
        scores += mask.unsqueeze(0).unsqueeze(1)  # [batch, heads, seq, seq]

        # 计算注意力权重
        attention = torch.softmax(scores, dim=-1)

        # 多头注意力融合
        output = torch.matmul(attention, values)  # [batch, heads, seq, dim]
        output = torch.einsum('bhqd,hdk->bhqk', output, self.W_o)  # 多头权重融合
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)

        return self.fc_out(output)

class TransformerEncoderLayerWithResidual(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, max_window=15, min_window=5):
        super().__init__()
        self.self_attn = DynamicWindowSparseAttention(d_model, nhead, max_window=max_window, min_window=min_window)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

class xLSTMBlockWithResidual(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, lstm_type):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden_state):
        output, hidden_state = self.lstm(x, hidden_state)
        if self.lstm.bidirectional:
            output = output[:, :, :self.lstm.hidden_size] + output[:, :, self.lstm.hidden_size:]
        output = self.layer_norm(output)
        output = self.dropout(output)
        return x + output, hidden_state

############################################
# 完整模型
############################################
class xLSTMWithTransformer(nn.Module):
    def __init__(self, input_size=1, output_size=1, embedding_size=64,
                 hidden_size=64, num_layers=2, num_blocks=4, dropout=0.1,
                 bidirectional=True, num_heads=4, gat_hidden=32, num_nodes=4):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        # 改进点：替换GAT为ST-GAT
        self.stgat = STGAT(in_channels=input_size, out_channels=gat_hidden)

        # 特征融合（调整输入维度）
        self.feature_fusion = nn.Sequential(
            nn.Linear(input_size + gat_hidden, embedding_size),
            nn.ReLU(),
            nn.LayerNorm(embedding_size)
        )
        self.embed_dropout = nn.Dropout(0.2)

        # LSTM模块（保持不变）
        self.blocks = nn.ModuleList([
            xLSTMBlockWithResidual(
                embedding_size if i == 0 else hidden_size,
                hidden_size,
                num_layers,
                dropout,
                bidirectional,
                "slstm"
            ) for i in range(self.num_blocks)
        ])

        # Transformer模块（保持不变）
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerWithResidual(hidden_size, num_heads, max_window=20, min_window=5, dropout=dropout)
            for _ in range(self.num_blocks)
        ])

        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_seq, edge_index):
        batch_size, seq_len, num_nodes, _ = input_seq.shape
        gat_features = []
        for t in range(seq_len):
            node_feats = input_seq[:, t]  # [batch, num_nodes, 1]
            batch_nodes = node_feats.reshape(-1, 1)
            # 构造批次图结构
            batch_edge_index = []
            for i in range(batch_size):
                batch_edge_index.append(edge_index + i * num_nodes)
            batch_edge_index = torch.cat(batch_edge_index, dim=1)

            # 改进点：使用ST-GAT替代原始GAT
            stgat_out = self.stgat(batch_nodes, batch_edge_index)  # [batch*num_nodes, gat_hidden]
            stgat_out = stgat_out.view(batch_size, num_nodes, -1)  # [batch, num_nodes, gat_hidden]
            gat_features.append(stgat_out)

        # 后续处理保持不变
        gat_features = torch.stack(gat_features, dim=1)
        combined = torch.cat([input_seq, gat_features], dim=-1)
        fused = self.embed_dropout(self.feature_fusion(combined))
        output_seq = fused.mean(dim=2)
        output_seq = output_seq.permute(1, 0, 2)

        for i in range(self.num_blocks):
            output_seq, _ = self.blocks[i](output_seq, None)
            output_seq = self.transformer_layers[i](output_seq)

        final_output = self.dropout_layer(output_seq[-1])
        return self.output_layer(final_output)

############################################
# 数据加载与预处理
############################################
def load_graph_data(file_path):
    adj_matrix = pd.read_csv(file_path, index_col=0).values
    edge_indices = []
    for i in range(4):
        for j in range(4):
            # 同时加入自环（对许多GAT实现有帮助）
            if adj_matrix[i][j] > 0 or i == j:
                edge_indices.append([i, j])
    return torch.tensor(edge_indices, dtype=torch.long).t()

if __name__ == "__main__":
    # 加载客流数据（附件2）
    data = pd.read_csv('20sshanghai.csv', index_col=0)
    # X：包含F1,F2,F3,F4四个区域数据；Y：预测F1
    X = data[['F1', 'F2', 'F3', 'F4']].values.reshape(-1, 4, 1)  # [samples, nodes, feature]
    Y = data[['F1']].values
    # 加载图数据（附件3）
    edge_index = load_graph_data('graph.csv')

    # 数据归一化
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    x = X_scaler.fit_transform(X.reshape(-1, 1)).reshape(-1, 4, 1)
    y = Y_scaler.fit_transform(Y)

    # 构造时间窗口（例如窗口长度为7）
    def create_windows(x, y, window_size=7):
        xs, ys = [], []
        for i in range(len(x) - window_size):
            xs.append(x[i:i + window_size])
            ys.append(y[i + window_size])
        return np.array(xs), np.array(ys)

    x_windows, y_windows = create_windows(x, y)
    x_windows = x_windows.reshape(-1, 7, 4, 1)  # [samples, seq_len, num_nodes, feature]

    # 划分训练集与测试集（保持时间顺序）
    x_train, x_test, y_train, y_test = train_test_split(x_windows, y_windows, test_size=0.2, shuffle=False)

    # 转换为Tensor
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # 数据加载器
    train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=128, shuffle=True)

    # 初始化模型及优化器（适当提高学习率并降低权重衰减）
    model = xLSTMWithTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, edge_index)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test_tensor, edge_index)
            test_loss = criterion(test_outputs, y_test_tensor)
        print(f"Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f}, Test Loss {test_loss:.4f}")

    # 评估并可视化
    with torch.no_grad():
        y_pred = model(x_test_tensor, edge_index).numpy()
        y_pred = Y_scaler.inverse_transform(y_pred)
        y_test_inv = Y_scaler.inverse_transform(y_test)
    print("\nMetrics:")
    print(f"MSE: {mean_squared_error(y_test_inv, y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(y_test_inv, y_pred):.3f}")
    print(f"R²: {r2_score(y_test_inv, y_pred):.3f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test_inv, y_pred) * 100:.2f}%")

    plt.figure(figsize=(12,6))
    plt.plot(y_test_inv, label='True')
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.legend()
    plt.title("Prediction Results")
    plt.show()
