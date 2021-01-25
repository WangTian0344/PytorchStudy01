import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
# 网络的保存和加载

torch.manual_seed(1)        # 设置随机数种子

# 假数据
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.__pow__(2)
x, y = Variable(x,requires_grad=False),Variable(y,requires_grad=False)

# 生成网络，保存网络
def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    optimizer = torch.optim.SGD(net1.parameters(),lr=0.5)
    loss_func = torch.nn.MSELoss()

    # 训练
    for i in range(100):
        prediction = net1(x)
        loss = loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 保存整个网络
    torch.save(net1,"../net/net1.pkl")
    # 保存网络参数
    torch.save(net1.state_dict(),"../net/net1_params.pkl")

    return prediction

# 加载整个网络
def load_net():
    # 读取整个网络
    net2 = torch.load("../net/net1.pkl")
    prediction = net2(x)
    return prediction

# 加载网络参数
def load_param():
    # 新建网络
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    # 导入参数
    net3.load_state_dict(torch.load("../net/net1_params.pkl"))
    prediction = net3(x)
    return prediction


prediction_net1 = save()
prediction_net2 = load_net()
prediction_net3 = load_param()

# 绘制图像进行比较
plt.subplot(131)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction_net1.data.numpy(), 'r-', lw=5)

plt.subplot(132)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction_net2.data.numpy(), 'g-', lw=5)

plt.subplot(133)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction_net3.data.numpy(), 'b-', lw=5)

plt.show()




