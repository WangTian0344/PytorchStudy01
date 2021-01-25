# MNIST手写数据识别
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision

# 卷积神经网络
# 流程：卷积->激励->池化，向下采样->（重复前面过程）->展平多维的卷积图->全连接层->输出
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,      # 输入数据通道数
                out_channels=16,    # 卷积通道数
                kernel_size=5,      # 卷积核尺寸
                stride=1,           # 步长
                padding=2           # 控制zero-padding的数目（补0）
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 池化 最大池化窗口为2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32*7*7,10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)   # 展平多维的卷积图
        output = self.out(x)
        return output

cnn = CNN()
# print(cnn)

torch.manual_seed(1)        # 随机数种子

EPOCH = 1               # 训练批次
BATCH_SIZE = 50         # 数据个数
LR = 0.001              # 学习率
DOWNLOAD_MNIST = False  # 是否下载MNIST数据集，True下载，False不下载

# 加载训练数据
train_data = torchvision.datasets.MNIST(
    root="./mnist/",                                # 指定文件夹
    train=True,                                     # 用作训练数据
    transform=torchvision.transforms.ToTensor(),    # 将数据类型转换为torch.FloatTensor
    download=DOWNLOAD_MNIST                         # 是否下载数据集
)
# 加载测试数据
test_data = torchvision.datasets.MNIST(
    root="./mnist/",                                # 指定文件夹
    train=False,                                    # 不是训练数据
)
# 将训练数据放入Dataloader
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# 节约时间，只测试前2000个
test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1),volatile=True).type(torch.FloatTensor)
test_y = test_data.test_labels[:2000]
# Adam优化器
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
# 损失函数
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step ,(x,y) in enumerate(train_loader):
        # 从dataloader中读取数据并变量化
        b_x = Variable(x)
        b_y = Variable(y)
        # 训练
        output = cnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# 将测试数据放入训练好的网络中，得到预测结果
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
# 预测结果与实际数据的比较
print("{} prediction number".format(pred_y))
print("{} real number".format(test_y[:10].numpy()))

