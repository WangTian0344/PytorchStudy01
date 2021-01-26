import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision

class myRNN(nn.Module):
    def __init__(self):
        super(myRNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64,10)

    def forward(self,x):
        r_out, (h_n,h_c) = self.rnn(x,None)
        out = self(r_out[:,-1,:])
        return out

rnn = myRNN()
# print(rnn)

torch.manual_seed(1)        # 随机数种子

EPOCH = 1               # 训练批次
BATCH_SIZE = 50         # 数据个数
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.001              # 学习率
DOWNLOAD_MNIST = False   # 是否下载MNIST数据集，True下载，False不下载

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
test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]

optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step , (x,y) in enumerate(train_loader):
        b_x = Variable(x.view(-1,28,28))
        b_y = Variable(y)

        output = rnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

test_output = rnn(test_x[:10].view(-1,28,28))
pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
print("{} prediction number".format(pred_y))
print("{} real number".format(test_y[:10]))