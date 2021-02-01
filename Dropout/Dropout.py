# Dropout 防止过拟合
# 使用torch.nn中的Dropout函数舍弃部分值，防止过拟合

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)
N_SAMPLES = 20
N_HIDDEN = 300
# 训练数据
x = torch.unsqueeze(torch.linspace(-1,1,N_SAMPLES),1)
y = x + 0.3*torch.normal(torch.zeros(N_SAMPLES,1),torch.ones(N_SAMPLES,1))
x ,y = Variable(x,requires_grad=False),Variable(y,requires_grad=False)
# 测试数据
test_x = torch.unsqueeze(torch.linspace(-1,1,N_SAMPLES),1)
test_y = test_x + 0.3*torch.normal(torch.zeros(N_SAMPLES,1),torch.ones(N_SAMPLES,1))
test_x ,test_y = Variable(test_x,requires_grad=False),Variable(test_y,requires_grad=False)
# 显示训练和测试数据
plt.scatter(x.data.numpy(),y.data.numpy(),c="magenta",s=50,alpha=0.5,label="train")
plt.scatter(test_x.data.numpy(),test_y.data.numpy(),c="cyan",s=50,alpha=0.5,label="test")
plt.legend(loc="upper left")
plt.ylim((-2.5,2.5))
plt.show()
# 没有dropout的网络
net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1,N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,1),
)
# 有dropout的网络
net_dropout = torch.nn.Sequential(
    torch.nn.Linear(1,N_HIDDEN),
    torch.nn.Dropout(0.5),      # 舍弃50%的数据
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,N_HIDDEN),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,1),
)

optimizer_overfitting = torch.optim.Adam(net_overfitting.parameters(),lr=0.01)
optimizer_dropout = torch.optim.Adam(net_dropout.parameters(),lr=0.01)
loss_func = torch.nn.MSELoss()
# 训练
for i in range(500):
    pred_overfitting = net_overfitting(x)
    pred_dropout = net_dropout(x)

    loss_overfitting = loss_func(pred_overfitting,y)
    loss_dropout = loss_func(pred_dropout,y)

    optimizer_overfitting.zero_grad()
    optimizer_dropout.zero_grad()
    loss_overfitting.backward()
    loss_dropout.backward()
    optimizer_overfitting.step()
    optimizer_dropout.step()

    if i%10 == 0:
        # 将网络从训练模式改为测试模式
        net_overfitting.eval()
        net_dropout.eval()      # dropout网络训练和测试参数不同

        # 绘制测试曲线
        plt.cla()
        # 测试数据
        test_pred_ofit = net_overfitting(test_x)
        test_pred_drop = net_dropout(test_x)
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_ofit, test_y).data.numpy(),fontdict={'size': 20, 'color': 'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data.numpy(),fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)

        # 将网络从测试模式改为训练模式
        net_overfitting.train()
        net_dropout.train()
plt.ioff()
plt.show()


