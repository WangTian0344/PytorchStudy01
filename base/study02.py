import torch
# torch 基本运算
data = torch.FloatTensor([1,-8,3,7])

print("data.abs:{}".format(torch.abs(data)))
print("data.sin:{}".format(torch.sin(data)))

data2 = torch.FloatTensor([[1,2],[2,5]])
print(data2)

print("data2*data2={}".format(torch.mm(data2,data2)))