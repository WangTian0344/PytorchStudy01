# torch DataLoader的使用
import torch
import torch.utils.data as Data

torch.manual_seed(1)        # 随机数种子
BATCH_SIZE = 5          # 批训练的数据个数

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

# 旧版写法
# torch_dataset = Data.TensorDataset(data_tensor=x,target_tensor=y)
# 新版写法
torch_dataset = Data.TensorDataset(x,y)

# 把dataset放入DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,           # 不打乱数据
    # num_workers=2         # 多线程，但是我的电脑cuda内存不足，运行时报错
)

for epoch in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):
        # 训练
        # ......
        print("Epoch:{}|Step:{}|batch x:{}|batch y:{}".format(epoch,step,batch_x.data.numpy(),batch_y.data.numpy()))
