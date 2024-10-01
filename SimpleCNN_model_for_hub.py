import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义简单的卷积神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        # 调用父类的构造函数
        super(SimpleCNN, self).__init__()

        # 第一个卷积层：输入通道数为1（灰度图像），输出通道数为20，卷积核大小为5x5，步长为1
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)

        # 第二个卷积层：输入通道数为20，输出通道数为50，卷积核大小为5x5，步长为1
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)

        # 全连接层1：输入为 4*4*50 个神经元（经过卷积和池化后的特征图大小），输出为500个神经元
        self.fc1 = nn.Linear(4 * 4 * 50, 500)

        # 全连接层2：输入为500个神经元，输出为分类数（MNIST有10类）
        self.fc2 = nn.Linear(500, num_classes)

    # 定义前向传播的逻辑
    def forward(self, x):
        # 第一层卷积：使用ReLU激活函数
        x = torch.relu(self.conv1(x))

        # 最大池化层：池化窗口大小为2x2，步长为2
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        # 第二层卷积：使用ReLU激活函数
        x = torch.relu(self.conv2(x))

        # 第二个最大池化层：池化窗口大小为2x2，步长为2
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        # 将特征图展平为向量：通过view函数将x转换为形状为（batch_size, 4*4*50）的张量
        x = x.view(-1, 4 * 4 * 50)

        # 全连接层1：使用ReLU激活函数
        x = torch.relu(self.fc1(x))

        # 全连接层2：输出分类结果
        x = self.fc2(x)

        return x


# 定义数据的预处理，包括将图像转换为张量并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化：均值为0.5，标准差为0.5
])

# 加载MNIST训练数据集，并应用预处理操作
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# 使用DataLoader创建数据加载器，设置批量大小为64，且数据打乱顺序
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化模型，指定分类数为10（对应MNIST数据集的10类数字）
model = SimpleCNN(num_classes=10)

# 定义优化器，使用Adam优化算法，学习率为0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义损失函数，使用交叉熵损失函数（适用于分类问题）
criterion = nn.CrossEntropyLoss()

# 初始化TensorBoard的SummaryWriter，用于记录训练过程中的损失
writer = SummaryWriter('runs/experiment_1')

# 定义训练轮数
num_epochs = 5

# 开始训练模型
for epoch in range(num_epochs):
    # 遍历每个批次的训练数据
    for i, (images, labels) in enumerate(train_loader):
        # 前向传播：将输入图像传入模型，得到预测输出
        outputs = model(images)

        # 计算损失：比较预测输出和真实标签
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()  # 清除上一次的梯度
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数

        # 每个批次结束后，将当前损失记录到TensorBoard中
        writer.add_scalar('Training/Loss', loss.item(), epoch * len(train_loader) + i)

        # 每训练100个批次，打印一次当前的训练状态
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 训练结束后，保存模型的参数到文件中
torch.save(model.state_dict(), 'model_trained.pth')
print("Model has been saved as 'model_trained.pth'")

# 关闭SummaryWriter，结束TensorBoard的记录
writer.close()
