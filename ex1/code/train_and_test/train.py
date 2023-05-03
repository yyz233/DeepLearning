import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt


# 绘制随epoch变化的图像
def draw(acc_label, loss_label):
    acc_label = [float(item) for item in acc_label]
    loss_label= [float(item) for item in loss_label]
    plt.plot(acc_label, color='r')
    plt.show()
    plt.plot(loss_label, color='b')
    plt.show()


class Train:

    def __init__(self, model, input_size, hidden_size, num_classes, batch_size, learning_rate, epochs):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model(input_size, hidden_size, num_classes).to(self.device)
        self.model.train().to(self.device)
        self.batch_size = batch_size
        self.epochs = epochs

        train_dataset = torchvision.datasets.MNIST(
            root="./data/",
            train=True,
            transform=transforms.ToTensor(),
            download=True
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            # 多线程载入，不让CPU卡脖子
            num_workers=4,
            shuffle=True
        )

        test_dataset = torchvision.datasets.MNIST(
            root="./data/",
            train=False,
            transform=transforms.ToTensor(),
            download=False
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=100,
            # 多线程载入，不让CPU卡脖子
            num_workers=4,
            shuffle=False
        )

        # 定义损失函数、优化器
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self):
        start = time.time()
        loss_label = []
        acc_label = []
        for epoch in range(self.epochs):
            correct = 0
            total = 0
            loss_now = 0
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.reshape(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)
                # 前向传播
                outputs = self.model(images).to(self.device)
                loss = self.criterion(outputs, labels).to(self.device)
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 统计准确率和最新的loss
                loss_now = loss
                _, predicted = torch.max(outputs, 1)  # max函数返回最大值和索引的元组，我们仅需用到索引值作为标签
                correct += (predicted == labels).sum().item()
                total += predicted.size(0)
            now = time.time()
            acc_label.append(correct/total)
            loss_label.append(loss_now)
            print('epoch ' + str(epoch) + ': Accuracy on train dataset '+str(correct/total) +
                  ', loss on train dataset '+str(loss_now.item()))
            print('lasts:' + str(now-start) + ' s')
        # 绘制随epoch变化的图像
        draw(acc_label, loss_label)

    def test(self):
        test_model = self.model
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.reshape(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)
                outputs = test_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of ' + str(self.model) + ' on test images: {} %'.format(100 * correct / total))

    def save(self):
        torch.save(self.model, './save_model/'+str(self.model)+'.ckpt')
