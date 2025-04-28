import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from PIL import Image

# 设置随机种子以确保结果可复现
torch.manual_seed(42)

# 定义超参数
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = 32  # CIFAR-10图像大小为32x32

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 定义CNN模型
class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)  # CIFAR-10有10个类别
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 初始化模型
model = CIFAR10Classifier()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# 评估函数
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')

if __name__ == '__main__':
    print("正在下载CIFAR-10数据集...")
    # 下载训练集
    train_dataset = datasets.CIFAR10(root='./data', 
                                   train=True,
                                   download=True,
                                   transform=transform)
    
    # 下载测试集
    test_dataset = datasets.CIFAR10(root='./data',
                                  train=False,
                                  download=True,
                                  transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, 
                            batch_size=BATCH_SIZE,
                            shuffle=True)
    
    test_loader = DataLoader(test_dataset,
                           batch_size=BATCH_SIZE,
                           shuffle=False)
    
    print("数据集下载完成！")
    print("开始训练模型...")
    train_model(model, train_loader, criterion, optimizer, EPOCHS)
    
    print("评估模型...")
    evaluate_model(model, test_loader) 