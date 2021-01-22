import numpy as np
import torch
# numpy 和 torch 数据互相转换
numpy_Data = np.arange(6).reshape((2,3))
torch_Data = torch.from_numpy(numpy_Data)
tensor2array = torch_Data.numpy()

print("numpy_data:\n{}".format(numpy_Data))
print("torch_Data:\n{}".format(torch_Data))
print("tensor2array:\n{}".format(tensor2array))