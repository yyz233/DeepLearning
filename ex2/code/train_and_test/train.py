import torchvision
import torch
import torchvision.transforms as transforms
from model.model import AlexNet
import torch.nn as nn
import time
from tensorboardX import SummaryWriter


class Train:

    def __init__(self, epoch, batch_size):
        self.writer = SummaryWriter('./result')
        self.device = self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.batch_size = batch_size
        self.epoch = epoch
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False
        )
        self.model = AlexNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        start = time.time()
        now_iter = 0
        for epoch in range(self.epoch):
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.train_loader):
                now_iter += 1
                images = images.to(self.device)
                labels = labels.to(self.device)
                # 前向传播
                outputs = self.model(images).to(self.device)
                loss = self.criterion(outputs, labels).to(self.device)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('loss', loss.item(), now_iter)
                # 统计准确率和最新的loss
                _, predicted = torch.max(outputs, 1)  # max函数返回最大值和索引的元组，我们仅需用到索引值作为标签
                correct += (predicted == labels).sum().item()
                total += predicted.size(0)
            now = time.time()
            self.writer.add_scalar('accuracy', correct / total, epoch+1)
            print('epoch ' + str(epoch) + ': Accuracy on train dataset ' + str(correct / total))
            print('lasts:' + str(now - start) + ' s')

    def test(self):
        test_model = self.model
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = test_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of ' + str(self.model) + ' on test images: {} %'.format(100 * correct / total))

    def save(self):
        torch.save(self.model, './save_model/'+str(self.model)+'.ckpt')
