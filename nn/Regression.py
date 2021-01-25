# 一元二次方程回归
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as f
class myNet(torch.nn.Module):  # 继承torch的Module
    def __init__(self,n_feature,n_hidden,n_output):
        super(myNet,self).__init__() # 继承super的init功能
        self.hidden = torch.nn.Linear(n_feature,n_hidden) # 隐藏层线性输出
        self.pridect = torch.nn.Linear(n_hidden,n_output) # 输出层线性输出

    def forward(self,x): # 重写Module中的forward功能
        # 正向传播输入值, 神经网络分析出输出值
        x = f.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        x = self.pridect(x)         # 输出值
        return x
net = myNet(n_feature=1,n_hidden=10,n_output=1)
# print(net)


# 随机生成y=x^2+b 并用Variable修饰
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.__pow__(2)+0.2*torch.rand(x.size())
x,y = Variable(x),Variable(y)

"""绘制x和y的图像
plt.scatter(x.data.numpy(),y.data.numpy())
plt.show()
"""

optimizer = torch.optim.SGD(net.parameters(),lr=0.5) # 传入net所有参数，学习率维0.5
loss_Func = torch.nn.MSELoss() # 预测值和真实值的误差函数——均方差

for i in range(100):
    prediction = net(x) # 计算预测值
    loss = loss_Func(prediction,y) # 计算预测值和实际值的误差
    optimizer.zero_grad() # 清空上一步的残余更新数值
    loss.backward() # 误差反向传播，计算参数更新值
    optimizer.step() # 将参数更新值施加到 net 的 parameters 上
    # 可视化训练过程
    if i % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, "Loss=(%0.4f)" %(loss.data.numpy()))
        plt.pause(0.5)
