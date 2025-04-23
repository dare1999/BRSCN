import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from block import xLSTMBlock
from thop import profile

#设置随机数种子
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#实例化模型
class xLSTM(nn.Module):
    def __init__(self, out_putsize, embedding_size, hidden_size, num_layers, num_blocks,
                 dropout=0.0, bidirectional=True, lstm_type="slstm"):
        super(xLSTM, self).__init__()
        self.out_putsize = out_putsize
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm_type = lstm_type

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.blocks = nn.ModuleList([xLSTMBlock(embedding_size if i == 0 else hidden_size,
                                                hidden_size, num_layers, dropout, bidirectional, lstm_type)
                                     for i in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, input_seq, hidden_states=None):
        # embedded_seq = self.embedding(input_seq)

        if hidden_states is None:
            hidden_states = [None] * self.num_blocks

        output_seq = input_seq
        for i, block in enumerate(self.blocks):
            output_seq, hidden_state = block(output_seq, hidden_states[i])
            if self.lstm_type == "slstm":
                hidden_states[i] = [[hidden_state[j][0], hidden_state[j][1]] for j in range(len(hidden_state))]
            else:
                hidden_states[i] = hidden_state

        output_seq = self.output_layer(output_seq)
        out = torch.relu(self.fc1(output_seq[:,-1,:]))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))

        return out

#定义模型参数
output_size = 1
embedding_size = 4
hidden_size = 4
num_layers = 2
num_blocks = 4
dropout = 0.1
bidirectional = False
lstm_type = "slstm"

#实例化模型
model = xLSTM(output_size, embedding_size, hidden_size, num_layers, num_blocks, dropout, bidirectional, lstm_type)

#读取数据
data = pd.read_csv('20sshanghai.csv',index_col=0)#20s.csvshanghai.csv

#划分X和Y
X = data[['F1','F2','F3','F4']].values
Y = data[['F1']].values

#数据归一化
X_Scaler = MinMaxScaler()
Y_Scaler = MinMaxScaler()

x = X_Scaler.fit_transform(X)
y = Y_Scaler.fit_transform(Y)

#定义时间滑窗函数
def create_windows(x,y,windows=7):#7
    xs = []
    ys = []
    for i in range(len(x)-windows):
        xs.append(x[i:i+windows,:])
        ys.append(y[i+windows])
    return np.array(xs),np.array(ys)

#构造时间滑窗数据
x, y = create_windows(x,y)

# 定义训练参数
learning_rate = 0.001
batch_size = 512#512
num_epochs = 600

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# 转换为Tensor数据
x_train_tensor = torch.Tensor(x_train)
y_train_tensor = torch.Tensor(y_train)
x_test_tensor = torch.Tensor(x_test)
y_test_tensor = torch.Tensor(y_test)

# 创建数据加载器
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
print(f'{total_params / (1024 * 1024):.2f}M total parameters.')
macs, params = profile(model, inputs=(train_dataset[:1][0],), verbose=False)
print(f"FLOPs (浮点运算次数): {macs}")
print(f"参数量: {params}")

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 将模型移动到指定的设备
device = 'cpu'
model = model.to(device)

# 存储训练和测试过程中的损失值
train_loss_history = []
test_loss_history = []

# 开始训练
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    total_loss = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()  # 梯度清零

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
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        x_test_tensor = x_test_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)
        test_outputs = model(x_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        test_loss_history.append(test_loss.item())
        # 打印训练和测试过程中的损失值和评价指标
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Test Loss: {test_loss.item():.4f}")

#数据反归一化
with torch.no_grad():
    y_pred = model(x_test_tensor).numpy()
    y_pred = Y_Scaler.inverse_transform(y_pred)
    y_test = Y_Scaler.inverse_transform(y_test)

#计算评价指标
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
mape = mean_absolute_percentage_error(y_test,y_pred)
#打印指标
print('MSE:',mse)
print('MAE:',mae)
print('R2:',r2)
print('MAPE:',mape)
print('RSME:',mse**0.5)

# 可视化损失曲线
plt.figure(figsize=(10,5),dpi=300)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(ls='--')
plt.legend()
plt.savefig('figure/loss',dpi=300,bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,5),dpi=300)
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.grid(ls='--')
plt.legend()
plt.savefig('figure/true vs pre',dpi=300,bbox_inches='tight')
plt.show()

y_test = y_test.flatten()
y_pred = y_pred.flatten()
results = pd.DataFrame({
    'True Values': y_test,
    'Predicted Values': y_pred
})

# 保存到Excel文件
results.to_excel('prediction_results.xlsx', index=False)