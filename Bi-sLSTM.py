import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# 尝试导入 thop，如果失败则跳过 FLOPs 计算
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    print("警告：未找到 'thop' 模块，将跳过 FLOPs 和参数量计算。请安装 thop：pip install thop")
    THOP_AVAILABLE = False

# 设置随机数种子
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# sLSTMCell 实现
class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), dim=1)
        i = torch.sigmoid(self.W_i(combined))
        f = torch.sigmoid(self.W_f(combined))
        c_tilde = torch.tanh(self.W_c(combined))
        o = torch.sigmoid(self.W_o(combined))
        c = f * c_prev + i * c_tilde
        h = o * torch.tanh(c)
        return h, c

# sLSTM 实现
class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else (hidden_size * 2 if bidirectional else hidden_size)
            self.layers.append(sLSTMCell(in_size, hidden_size))
            if bidirectional:
                self.layers.append(sLSTMCell(in_size, hidden_size))

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        num_directions = 2 if self.bidirectional else 1

        if hidden is None:
            h_t = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = hidden

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_layer, c_layer = [], []
            for layer in range(self.num_layers):
                for direction in range(num_directions):
                    idx = layer * num_directions + direction
                    h_prev = h_t[idx]
                    c_prev = c_t[idx]
                    h, c = self.layers[idx](x_t, (h_prev, c_prev))
                    if self.dropout and layer < self.num_layers - 1:
                        h = self.dropout(h)
                    if direction == 0:
                        h_fwd = h
                    else:
                        h_bwd = h
                    h_layer.append(h)
                    c_layer.append(c)
                x_t = torch.cat((h_fwd, h_bwd), dim=1) if self.bidirectional else h_fwd
            h_t = torch.stack(h_layer)
            c_t = torch.stack(c_layer)
            outputs.append(x_t)
        output = torch.stack(outputs, dim=1)
        hidden = (h_t, c_t)
        return output, hidden

# xLSTMBlock 实现
class xLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, lstm_type):
        super().__init__()
        if lstm_type == "slstm":
            self.lstm = sLSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.layer_norm = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, hidden_state):
        output, hidden_state = self.lstm(x, hidden_state)
        if self.dropout:
            output = self.dropout(output)
        output = self.layer_norm(output)
        return output, hidden_state

# 实例化模型
class xLSTM(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, num_blocks,
                 dropout=0.0, bidirectional=True, lstm_type="slstm"):
        super(xLSTM, self).__init__()
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm_type = lstm_type

        self.blocks = nn.ModuleList([
            xLSTMBlock(embedding_size if i == 0 else hidden_size * (2 if bidirectional else 1),
                       hidden_size, num_layers, dropout, bidirectional, lstm_type)
            for i in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)

    def forward(self, input_seq, hidden_states=None):
        if hidden_states is None:
            hidden_states = [None] * self.num_blocks

        output_seq = input_seq
        for i, block in enumerate(self.blocks):
            output_seq, hidden_state = block(output_seq, hidden_states[i])
            if self.lstm_type == "slstm":
                h_t, c_t = hidden_state
                hidden_states[i] = [(h_t[j], c_t[j]) for j in range(h_t.size(0))]
            else:
                hidden_states[i] = hidden_state

        output_seq = self.output_layer(output_seq[:, -1, :])
        out = torch.relu(self.fc1(output_seq))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))

        return out

# 定义模型参数
output_size = 1
embedding_size = 4
hidden_size = 16
num_layers = 1
num_blocks = 2
dropout = 0.1
bidirectional = True  # 启用双向
lstm_type = "slstm"

# 实例化模型
model = xLSTM(output_size, embedding_size, hidden_size, num_layers, num_blocks, dropout, bidirectional, lstm_type)

# 读取数据
data = pd.read_csv('20sshanghai.csv', index_col=0)
print(data['F1'].describe())  # 打印 F1 的统计信息

# 划分 X 和 Y
X = data[['F1', 'F2', 'F3', 'F4']].values
Y = data[['F1']].values

# 数据归一化
X_Scaler = MinMaxScaler()
Y_Scaler = MinMaxScaler()
x = X_Scaler.fit_transform(X)
y = Y_Scaler.fit_transform(Y)

# 定义时间滑窗函数
def create_windows(x, y, windows=14):
    xs = []
    ys = []
    for i in range(len(x) - windows):
        xs.append(x[i:i + windows, :])
        ys.append(y[i + windows])
    return np.array(xs), np.array(ys)

# 构造时间滑窗数据
x, y = create_windows(x, y)

# 定义训练参数
learning_rate = 0.0001
batch_size = 128
num_epochs = 1000

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# 转换为 Tensor 数据
x_train_tensor = torch.Tensor(x_train)
y_train_tensor = torch.Tensor(y_train)
x_test_tensor = torch.Tensor(x_test)
y_test_tensor = torch.Tensor(y_test)

# 创建数据加载器
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 计算参数量和 FLOPs（如果 thop 可用）
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
print(f"{total_params / (1024 * 1024):.2f}M total parameters.")
if THOP_AVAILABLE:
    macs, params = profile(model, inputs=(x_train_tensor[:1],), verbose=False)
    print(f"FLOPs (浮点运算次数): {macs}")
    print(f"参数量: {params}")

# 定义损失函数和优化器
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 将模型移动到指定的设备
device = 'cpu'
model = model.to(device)

# 存储训练和测试过程中的损失值
train_loss_history = []
test_loss_history = []
best_loss = float('inf')
patience = 20
trigger_times = 0

# 开始训练
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_train_loss = total_loss / len(train_loader)
    train_loss_history.append(average_train_loss)

    model.eval()
    with torch.no_grad():
        x_test_tensor = x_test_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)
        test_outputs = model(x_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        test_loss_history.append(test_loss.item())
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Test Loss: {test_loss.item():.4f}")

    if average_train_loss < best_loss:
        best_loss = average_train_loss
        trigger_times = 0
    else:
        trigger_times += 1
    if trigger_times >= patience:
        print("Early stopping!")
        break

# 数据反归一化
with torch.no_grad():
    y_pred = model(x_test_tensor).numpy()
    y_pred = Y_Scaler.inverse_transform(y_pred)
    y_test = Y_Scaler.inverse_transform(y_test)
    print("y_pred range:", y_pred.min(), y_pred.max())
    print("y_test range:", y_test.min(), y_test.max())

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

# 创建保存图像的目录
os.makedirs('figure', exist_ok=True)

# 可视化损失曲线
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(ls='--')
plt.legend()
plt.savefig('figure/loss.png', dpi=300, bbox_inches='tight')
plt.show()

# 可视化真实值与预测值
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.grid(ls='--')
plt.legend()
plt.savefig('figure/true_vs_pred.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存结果到 Excel 文件
y_test = y_test.flatten()
y_pred = y_pred.flatten()
results = pd.DataFrame({
    'True Values': y_test,
    'Predicted Values': y_pred
})
results.to_excel('prediction_results.xlsx', index=False)