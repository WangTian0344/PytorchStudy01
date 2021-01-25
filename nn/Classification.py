# 简单分类网络
import torch
from torch.autograd import Variable
import torch.nn.functional as f
import matplotlib.pyplot as plt

class myNet(torch.nn.Module):  #  继承torch的Module
    def __init__(self,n_feature,n_hidden,n_output):
        super(myNet,self).__init__()                         # 继承init
        self.hidden = torch.nn.Linear(n_feature,n_hidden)    # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden,n_output)        # 输出层线性输出

    def forward(self,x):
        # 正向转播输入值，神经网络分析出输出值
        x = f.relu(self.hidden(x))      # 激励函数（隐藏层的线性值）
        x = self.out(x)                 # 输出值，但并不是预测值
        return x

net = myNet(n_feature=2,n_hidden=10,n_output=2)


# 假数据
n_data = torch.ones(100,2)      # shape = (100,2) 的全一张量
x0 = torch.normal(2*n_data,1)   # shape = (100,2) 的张量，根据normal生成
y0 = torch.zeros(100)           # shape = (100,1) 的全零张量
x1 = torch.normal(-2*n_data,1)  # shape = (100,2) 的张量，根据normal生成
y1 = torch.ones(100)            # shape = (100,1) 的全一张量

# 将x0和x1张量在第0维上连接，shape = (200,2)，数据类型Float
x = torch.cat((x0,x1),0).type(torch.FloatTensor)
# 将y0和y1张量在第1维上连接，shape = (200,1)，数据类型Long
y = torch.cat((y0,y1),).type(torch.LongTensor)
# 参数化
x,y = Variable(x),Variable(y)


# 训练工具
optimizer = torch.optim.SGD(net.parameters(),lr=0.02)   # 传入net所有参数，学习率0.02
loss_func = torch.nn.CrossEntropyLoss()                 # 损失函数

plt.ion()   # 画图
plt.show()

for i in range(100):
    out = net(x)              # 将x送入网络，计算输出值
    loss = loss_func(out,y)   # 计算误差
    optimizer.zero_grad()     # 清空上一步的参与更新参数值
    loss.backward()           # 误差反向传播，计算参数更新值
    optimizer.step()          # 将参数更新值施加到net的parameters中

    if i % 2 == 0:
        plt.cla()       # 清空当前坐标轴
        # 输出值经过激励函数softmax才是预测值
        prediction = torch.max(f.softmax(out),1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()  # 停止画图
plt.show()


