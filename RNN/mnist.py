# MNIST 数据集图片显示
import torchvision
import matplotlib.pyplot as plt
# 图片序号 0-59999
NUM = 59999
train_data = torchvision.datasets.MNIST(
    root="./mnist/",                                # 指定文件夹
    train=True,                                     # 用作训练数据
    transform=torchvision.transforms.ToTensor(),    # 将数据类型转换为torch.FloatTensor
    download=False                                  # 是否下载数据集
)

print(train_data.train_data.size())
print(train_data.test_labels.size())
plt.imshow(train_data.test_data[NUM].numpy(),cmap="gray")
plt.title("%i" % train_data.train_labels[NUM])
plt.show()