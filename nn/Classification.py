# 简单分类网络
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 假数据
n_data = torch.ones(100,1)      # shape = (100,2) 的全一张量
x0 = torch.normal(2*n_data,1)   # shape = (100,2) 的张量，根据normal生成
y0 = torch.zeros(100)           # shape = (100,1) 的全零张量
x1 = torch.normal(-2*n_data,1)  # shape = (100,2) 的张量，根据normal生成
y1 = torch.ones(100)            # shape = (100,1) 的全一张量

# 将x0和x1张量在第0维上连接，shape = (200,2)，数据类型Float
x = torch.cat((x0,x1),0).type(torch.FloatTensor)
# 将y0和y1张量在第1维上连接，shape = (200,1)，数据类型Long
y = torch.cat((y0,y1),).type(torch.LongTensor)

x,y = Variable(x),Variable(y)

plt.scatter(x.data.numpy(),y.data.numpy())
plt.show()

