import torch
import torch.nn.functional as F
# 两种搭建神经网络的方法

# 类继承方式搭建神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net1 = Net(1, 10, 1)


# 快速搭建神经网络
net2 = torch.nn.Sequential(torch.nn.Linear(1,10),
                           torch.nn.ReLU(),
                           torch.nn.Linear(10,1))

# 本质上net1和net2完全一样
print(net1)
print(net2)