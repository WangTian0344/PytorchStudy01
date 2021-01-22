# torch 激励函数

import torch
import torch.nn.functional as f
from torch.autograd import Variable
import matplotlib.pyplot as plt
# 生成数据
x = torch.linspace(-5,5,200)
x = Variable(x)
# 转化为numpy好画图
x_np = x.data.numpy()
# 计算激励函数值
y_relu = f.relu(x).data.numpy()
y_sigmoid = f.sigmoid(x).data.numpy()
y_tanh = f.tanh(x).data.numpy()
y_softplus = f.softplus(x).data.numpy()
# 画图
plt.figure(1,figsize=(8,6))
plt.subplot(221)
plt.plot(x_np,y_relu,c='red',label='relu')
plt.ylim((-1,5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()


