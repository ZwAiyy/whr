import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary

# 加载数据集转换成张量数据集
def create_dataset():
    train_data = CIFAR10(root='../data', train=True, transform=ToTensor())
    valid_dataset = CIFAR10(root='../data', train=False, transform=ToTensor())  # ToTensor(): 将图片数据转换成张量数据
    return train_data, valid_dataset

class ImgModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入图像为 32x32。两次卷积（无填充）后：
        # 卷积 1：3 -> 6，kernel=3，stride=1，padding=0。输出尺寸 = 32 - 3 + 1 = 30。
        # 池化 1：kernel=2，stride=2。输出尺寸 = 30 / 2 = 15。
        # 卷积 2：6 -> 16，kernel=3，stride=1，padding=0。输出尺寸 = 15 - 3 + 1 = 13。
        # 池化 2：kernel=2，stride=2。输出尺寸 = 13 // 2 = 6
        # 接着展平为 16 * 6 * 6 = 576
        # 输入3，输出6，卷积核大小3*3，步长1，填充0
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
        self.pool1 = nn.MaxPool2d(2, 2, 0)
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
        self.pool2 = nn.MaxPool2d(2, 2, 0)
        # 池化层
        self.linear1 = nn.Linear(16 * 6 * 6, 120)
        # 隐藏层
        self.linear2 = nn.Linear(120, 84)
        # 输出层
        # 10分类
        self.out = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))# 卷积+激活+池化
        x = self.pool2(torch.relu(self.conv2(x)))# 2 卷积+激活+池化
        x = x.reshape(shape=(x.shape[0], -1))
        x = torch.relu(self.linear1(x))
        # 第2层隐藏层
        x = torch.relu(self.linear2(x))
        # 输出层
        x = self.out(x)
        return x


BATCH_SIZE = 8  # 批量大小
EPOCHS = 1  # 训练轮数
LEARNING_RATE = 0.01  # 学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(train_dataset):
    # 加载数据
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 加载对象
    model = ImgModel()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        total_loss, total_samples, total_correct, start = 0.0, 0.0, 0.0, time.time()
        # 训练模式
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # 预测
            y_pred = model(x)
            # 计算损失
            loss = criterion(y_pred, y)
            # 清零
            optimizer.zero_grad()
            # 梯度计算
            loss.backward()
            # 参数更新
            optimizer.step()

            # 统计预测正确的样本个数
            total_correct += ((torch.argmax(y_pred, 1) == y).sum())  # 预测为true的数量
            # print(y_pred)
            # print(torch.argmax(y_pred, 1))

            # loss.item(): 当前批次平均损失值
            total_loss += loss.item() * len(y)  # 统计当前批次的总损失值

            total_samples += len(y)  # 统计当前批次的样本数
        end = time.time()
        print('轮数:%2s loss:%.5f acc:%.2f time:%.2fs' % (
            epoch + 1, total_loss / total_samples, total_correct / total_samples, end - start))
    # 保存训练模型
    torch.save(obj=model.state_dict(), f='../model/imagemodel.pth')


def test(test_dataset):
    pass

# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
# model = ImgModel()
# model.load_state_dict(torch.load('./model/imagemodel.pth'))

if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()
    # print('train_dataset.data.shape->', train_dataset.data.shape)
    # 图像
    # plt.figure(figsize=(2, 2))
    # plt.imshow(train_dataset.data[1])
    # plt.title(train_dataset.targets[1])
    # plt.show()
    model = ImgModel()
    # summary(model, input_size=(3, 32, 32))

    # Conv2d-1 (168个)：(输入通道3 * 卷积核高3 * 卷积核宽3 + 1个偏置) * 输出通道6 = (3*3*3 + 1)*6 = 168。
    # Conv2d-3 (880个)：(输入通道6 * 3*3 + 1) * 16 = (54+1)*16 = 880。
    # Linear-5 (69,240个)：(输入特征 16*6*6=576) * 输出特征120 + 120个偏置 = 576*120 + 120 = 69,240。
    # Linear-7 (850个)：84*10 + 10 = 850。
    train(train_dataset)
# test(test_dataset)