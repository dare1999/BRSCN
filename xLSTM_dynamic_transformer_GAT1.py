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
import math
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# 时间注意力层
class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        batch, seq_len, num_nodes, d_model = x.shape
        x = x.view(batch, seq_len * num_nodes, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attention = torch.softmax(scores, dim=-1)
        output = torch.bmm(attention, V)
        return output.view(batch, seq_len, num_nodes, d_model)

# STGAT 模块
class STGAT(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_kernel=3, heads=2, layers=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            gat = nn.ModuleList([
                gnn.GATConv(in_channels if _ == 0 else out_channels, out_channels // heads, heads=heads, dropout=0.3),
                gnn.GATConv(out_channels, out_channels, heads=1, concat=False, dropout=0.3)
            ])
            self.layers.append(gat)
        self.tcn = nn.Conv1d(out_channels, out_channels, kernel_size=temporal_kernel, padding="same")
        self.norm = nn.LayerNorm(out_channels)
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index):
        batch_size = x.shape[0] // 4
        residual = self.residual(x)
        for gat1, gat2 in self.layers:
            x = F.elu(gat1(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
            x = gat2(x, edge_index)
        x = x + residual
        x = x.view(batch_size, 4, -1).permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1).reshape(-1, x.shape[1])
        return self.norm(x)

# 动态窗口稀疏注意力机制
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
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.3, layer_norm_eps=1e-5, max_window=20, min_window=5):
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

# 完整模型（简化版）
class xLSTMWithTransformer(nn.Module):
    def __init__(self, input_size=1, temporal_size=2, output_size=1, embedding_size=32, hidden_size=32,
                 num_layers=1, num_blocks=1, dropout=0.3, bidirectional=True, num_heads=2, gat_hidden=16, num_nodes=4):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_nodes = num_nodes

        # STGAT
        self.stgat = STGAT(in_channels=input_size, out_channels=gat_hidden, layers=1)
        self.temporal_attention = TemporalAttention(gat_hidden)

        # 时间特征嵌入
        self.temporal_embed = nn.Linear(temporal_size, gat_hidden)

        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(input_size + gat_hidden + gat_hidden, embedding_size),
            nn.ReLU(),
            nn.LayerNorm(embedding_size)
        )
        self.embed_dropout = nn.Dropout(dropout)

        # LSTM 和 Transformer 模块
        self.blocks = nn.ModuleList([
            xLSTMBlockWithResidual(
                embedding_size if i == 0 else hidden_size,
                hidden_size,
                num_layers,
                dropout,
                bidirectional,
                "slstm"
            ) for i in range(num_blocks)
        ])
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerWithResidual(hidden_size, num_heads, max_window=20, min_window=5, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # 增强输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, input_seq, temporal_features, edge_index):
        batch_size, seq_len, num_nodes, _ = input_seq.shape
        temporal_features = temporal_features[:, :, None, :].expand(-1, seq_len, num_nodes, -1)
        temporal_embed = self.temporal_embed(temporal_features)

        gat_features = []
        for t in range(seq_len):
            node_feats = input_seq[:, t].reshape(-1, 1)
            batch_edge_index = torch.cat([edge_index + i * num_nodes for i in range(batch_size)], dim=1)
            stgat_out = self.stgat(node_feats, batch_edge_index)
            stgat_out = stgat_out.view(batch_size, num_nodes, -1)
            gat_features.append(stgat_out)
        gat_features = torch.stack(gat_features, dim=1)

        gat_features = self.temporal_attention(gat_features)

        combined = torch.cat([input_seq, gat_features, temporal_embed], dim=-1)
        fused = self.embed_dropout(self.feature_fusion(combined))
        output_seq = fused.mean(dim=2)
        output_seq = output_seq.permute(1, 0, 2)

        for i in range(self.num_blocks):
            output_seq, _ = self.blocks[i](output_seq, None)
            output_seq = self.transformer_layers[i](output_seq)

        final_output = output_seq[-1]
        return self.output_layer(final_output)

# 数据加载与预处理
def load_graph_data(file_path):
    adj_matrix = pd.read_csv(file_path, index_col=0).values
    if adj_matrix.shape != (4, 4):
        corr = data[['F1', 'F2', 'F3', 'F4']].corr().values
        adj_matrix = (corr > 0.5).astype(int)
    edge_indices = []
    for i in range(4):
        for j in range(4):
            if adj_matrix[i][j] > 0 or i == j:
                edge_indices.append([i, j])
    return torch.tensor(edge_indices, dtype=torch.long).t()

def add_temporal_features(data):
    try:
        data['timestamp'] = pd.to_datetime(data.index)
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    except:
        print("警告：使用滞后特征")
        for lag in [1, 2, 3, 7]:
            for col in ['F1', 'F2', 'F3', 'F4']:
                data[f'{col}_lag{lag}'] = data[col].shift(lag)
        data = data.dropna()
        data['day_of_week'] = np.random.randint(0, 7, len(data))
        data['hour'] = np.random.randint(0, 24, len(data))
    return data

def create_windows(x, y, temporal, window_size=7):
    xs, ys, ts = [], [], []
    for i in range(len(x) - window_size):
        xs.append(x[i:i + window_size])
        ys.append(y[i + window_size])
        ts.append(temporal[i:i + window_size])
    return np.array(xs), np.array(ys), np.array(ts)

if __name__ == "__main__":
    # 加载数据
    data = pd.read_csv('20sshanghai.csv', index_col=0)
    print("F1 统计信息：", data['F1'].describe())
    data = add_temporal_features(data)

    # 分离节点特征和时间特征
    node_features = data[['F1', 'F2', 'F3', 'F4']].values
    try:
        temporal_features = data[['day_of_week_sin', 'day_of_week_cos', 'hour_sin', 'hour_cos']].values
        temporal_size = 4
    except:
        temporal_features = data[[f'{col}_lag{lag}' for col in ['F1', 'F2', 'F3', 'F4'] for lag in [1, 2, 3, 7]]].values
        temporal_size = 16
    Y = np.log1p(data[['F1']].values)  # 对数变换
    edge_index = load_graph_data('graph.csv')

    # 归一化
    node_scaler = MinMaxScaler()
    temporal_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    node_scaled = node_scaler.fit_transform(node_features).reshape(-1, 4, 1)
    temporal_scaled = temporal_scaler.fit_transform(temporal_features)
    y = Y_scaler.fit_transform(Y)

    # 创建时间窗口
    window_size = 7
    x_windows, y_windows, t_windows = create_windows(node_scaled, y, temporal_scaled, window_size=window_size)
    x_windows = x_windows.reshape(-1, window_size, 4, 1)
    t_windows = t_windows.reshape(-1, window_size, temporal_scaled.shape[-1])

    # 划分训练集、验证集和测试集
    indices = np.arange(len(x_windows))
    x_train, x_test, y_train, y_test, t_train, t_test, idx_train, idx_test = train_test_split(
        x_windows, y_windows, t_windows, indices, test_size=0.2, shuffle=False
    )
    x_train, x_val, y_train, y_val, t_train, t_val, idx_train, idx_val = train_test_split(
        x_train, y_train, t_train, idx_train, test_size=0.1, shuffle=False
    )

    # 转换为张量
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)
    t_train_tensor = torch.FloatTensor(t_train)
    x_val_tensor = torch.FloatTensor(x_val)
    y_val_tensor = torch.FloatTensor(y_val)
    t_val_tensor = torch.FloatTensor(t_val)
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.FloatTensor(y_test)
    t_test_tensor = torch.FloatTensor(t_test)

    # 数据加载器
    train_dataset = TensorDataset(x_train_tensor, t_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, t_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # 初始化模型和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = xLSTMWithTransformer(input_size=1, temporal_size=temporal_size, embedding_size=32, hidden_size=32,
                                 num_layers=1, num_blocks=1, num_heads=2, gat_hidden=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    # 训练循环
    best_val_loss = float('inf')
    patience = 20
    counter = 0
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, temporal, labels in train_loader:
            inputs, temporal, labels = inputs.to(device), temporal.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, temporal, edge_index.to(device))
            loss = 0.5 * criterion_mse(outputs, labels) + 0.5 * criterion_l1(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, temporal, labels in val_loader:
                inputs, temporal, labels = inputs.to(device), temporal.to(device), labels.to(device)
                outputs = model(inputs, temporal, edge_index.to(device))
                val_loss += criterion_mse(outputs, labels).item()
        val_loss /= len(val_loader)

        with torch.no_grad():
            test_outputs = model(x_test_tensor.to(device), t_test_tensor.to(device), edge_index.to(device))
            test_loss = criterion_mse(test_outputs, y_test_tensor.to(device))
        print(f"Epoch {epoch + 1}: 训练损失 {avg_train_loss:.4f}, 验证损失 {val_loss:.4f}, 测试损失 {test_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("早停触发")
                break
        scheduler.step()

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

    # 评估
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_tensor.to(device), t_test_tensor.to(device), edge_index.to(device)).cpu().numpy()
        y_pred = np.expm1(Y_scaler.inverse_transform(y_pred))
        y_test_inv = np.expm1(Y_scaler.inverse_transform(y_test))
    print("\n评估指标：")
    print(f"MSE: {mean_squared_error(y_test_inv, y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(y_test_inv, y_pred):.3f}")
    print(f"R²: {r2_score(y_test_inv, y_pred):.3f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test_inv, y_pred) * 100:.2f}%")

    # 可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='真实值')
    plt.plot(y_pred, label='预测值', alpha=0.7)
    plt.legend()
    plt.title("预测结果")
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()