import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from block import xLSTMBlock
import torch.onnx
import math

# 设置随机数种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# 定义动态窗口稀疏注意力机制
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

# 修改Transformer编码器层以使用动态窗口稀疏注意力机制
class TransformerEncoderLayerWithResidual(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, max_window=120, min_window=24):
        super().__init__()
        self.self_attn = DynamicWindowSparseAttention(
            d_model, nhead,
            max_window=max_window,
            min_window=min_window
        )
        # 保持原有前馈网络结构
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # 自注意力部分
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络部分
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# 修改 xLSTMBlock 定义，使其使用双向LSTM并添加残差连接和层归一化
class xLSTMBlockWithResidual(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, lstm_type):
        super(xLSTMBlockWithResidual, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden_state):
        output, hidden_state = self.lstm(x, hidden_state)
        # 如果是双向LSTM，拼接两个方向的输出并进行残差连接
        if self.lstm.bidirectional:
            output = output[:, :, :self.lstm.hidden_size] + output[:, :, self.lstm.hidden_size:]
        # Apply layer normalization and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        # Add residual connection
        output = x + output
        return output, hidden_state


# 实例化模型
class xLSTMWithTransformer(nn.Module):
    def __init__(self, input_size, output_size, embedding_size, hidden_size, num_layers, num_blocks,
                 dropout=0.1, bidirectional=True, num_heads=4):
        super(xLSTMWithTransformer, self).__init__()
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_heads = num_heads

        self.embedding = nn.Linear(input_size, embedding_size)  # 使用Linear层进行嵌入
        self.blocks = nn.ModuleList([xLSTMBlockWithResidual(embedding_size if i == 0 else hidden_size,
                                                            hidden_size, num_layers, dropout, bidirectional, "slstm")
                                     for i in range(num_blocks)])
        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayerWithResidual(hidden_size, num_heads, dropout=dropout)
             for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq, hidden_states=None):
        if hidden_states is None:
            hidden_states = [None] * self.num_blocks

        # Embedding层
        embedded_seq = self.embedding(input_seq)

        output_seq = embedded_seq
        for i in range(self.num_blocks):
            # xLSTM块
            output_seq, hidden_states[i] = self.blocks[i](output_seq, hidden_states[i])
            # Transformer Encoder层
            output_seq = self.transformer_layers[i](output_seq)

        # 取序列最后一个时间步的输出
        final_output = output_seq[:, -1, :]
        final_output = self.dropout(final_output)
        # 输出层
        out = self.output_layer(final_output)
        return out


# 定义模型参数
input_size = 1  # 输入特征数
output_size = 1
embedding_size = 64
hidden_size = 64
num_layers = 2
num_blocks = 4
dropout = 0.1
bidirectional = True
num_heads = 4

# 实例化模型
model = xLSTMWithTransformer(input_size, output_size, embedding_size, hidden_size, num_layers, num_blocks,
                             dropout, bidirectional, num_heads)

# 读取数据
data = pd.read_csv('20sshanghai.csv', index_col=0)

# 划分X和Y
X = data[['F1']].values#, 'F2', 'F3', 'F4'
Y = data[['F1']].values

# 数据归一化
X_Scaler = MinMaxScaler()
Y_Scaler = MinMaxScaler()

x = X_Scaler.fit_transform(X)
y = Y_Scaler.fit_transform(Y)


# 定义时间滑窗函数
def create_windows(x, y, window_size=7):
    xs = []
    ys = []
    for i in range(len(x) - window_size):
        xs.append(x[i:i + window_size, :])
        ys.append(y[i + window_size])
    return np.array(xs), np.array(ys)


# 构造时间滑窗数据
x, y = create_windows(x, y)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# 转换为Tensor数据
x_train_tensor = torch.Tensor(x_train)
y_train_tensor = torch.Tensor(y_train)
x_test_tensor = torch.Tensor(x_test)
y_test_tensor = torch.Tensor(y_test)

# 创建数据加载器
batch_size = 512
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 将模型移动到指定的设备
device = 'cpu'
model = model.to(device)

# 存储训练和测试过程中的损失值
train_loss_history = []
test_loss_history = []

# 定义早停参数
best_loss = float('inf')
early_stop_patience = 300
early_stop_counter = 0
checkpoint_path = 'best_model.pth'

# 开始训练
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 计算平均训练损失
    average_train_loss = total_loss / len(train_loader)
    train_loss_history.append(average_train_loss)

    # 在测试集上进行评估
    model.eval()
    with torch.no_grad():
        x_test_tensor = x_test_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)
        test_outputs = model(x_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        test_loss_history.append(test_loss.item())
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Test Loss: {test_loss.item():.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), checkpoint_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

# 数据反归一化
with torch.no_grad():
    y_pred = model(x_test_tensor).cpu().numpy()
    y_pred = Y_Scaler.inverse_transform(y_pred)
    y_test = Y_Scaler.inverse_transform(y_test)

# 计算评价指标
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# 打印指标
print('MSE:', mse)
print('MAE:', mae)
print('R2:', r2)
print('MAPE:', mape)
print('RMSE:', mse ** 0.5)

# 可视化损失曲线
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(ls='--')
plt.legend()
plt.savefig('figure/loss', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5), dpi=300)
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.grid(ls='--')
plt.legend()
plt.savefig('figure/true_vs_pred', dpi=300, bbox_inches='tight')
plt.show()

#torch.onnx.export(model, x_test_tensor, "model.onnx", verbose=True)
y_test = y_test.flatten()
y_pred = y_pred.flatten()
results = pd.DataFrame({
    'True Values': y_test,
    'Predicted Values': y_pred
})

# 保存到Excel文件
results.to_excel('prediction_results.xlsx', index=False)