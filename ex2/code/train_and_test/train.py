import torch
import torchvision.transforms as transforms
from model.model import AlexNet
from torch.utils.data import Dataset
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
import cv2
import os


class MyDataset(Dataset):

    def __init__(self, data, label, transform):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.transform(self.data[item][:]), self.label[item]


def data_read(size):
    path = '.\\data\\caltech-101\\101_ObjectCategories\\101_ObjectCategories'
    image_dir_paths = next(os.walk(path))[1]  # 获取所有子目录路径
    data = []
    labels = []

    # 读取数据
    for i, image_dir_path in enumerate(image_dir_paths):
        image_names = next(os.walk(path + '\\' + image_dir_path))[2]
        for image_name in image_names:
            image = cv2.imread(path + '\\' + image_dir_path + '\\' + image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            data.append(image)
            labels.append(i)

    return data, labels


def data_process(batch_size, size):
    my_transform = transforms.Compose(
        [transforms.ToPILImage(),
         # transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    )
    data, labels = data_read(size)
    # 拆分训练集、测试集和验证集
    # print(len(data), len(labels))
    (train_test_data,  validate_data, train_test_label, validate_label) = train_test_split(
        data,
        labels,
        test_size=0.1,
        stratify=labels,
        random_state=42
    )
    # print(len(train_test_data), len(train_test_label))
    (train_data, test_data, train_label, test_label) = train_test_split(
        train_test_data,
        train_test_label,
        test_size=0.2,
        stratify=train_test_label,
        random_state=42
    )
    train_dataset = MyDataset(train_data, train_label, my_transform)
    test_dataset = MyDataset(test_data, test_label, my_transform)
    validate_dataset = MyDataset(validate_data, validate_label, my_transform)
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True
    ), torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True
    ), torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True
        )


class Train:

    def __init__(self, epoch, batch_size, size=65):
        self.train_loader, self.test_loader, self.validate_loader = data_process(batch_size, size)
        self.writer = SummaryWriter('./result')
        self.device = self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.epoch = epoch
        self.model = AlexNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        start = time.time()
        now_iter = 0
        for epoch in range(self.epoch):
            correct = 0
            total = 0
            total_loss = 0.0
            for i, (images, labels) in enumerate(self.train_loader):
                now_iter += 1
                # print(now_iter)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # 前向传播
                outputs = self.model(images).to(self.device)
                loss = self.criterion(outputs, labels).to(self.device)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 统计准确率和最新的loss
                _, predicted = torch.max(outputs, 1)  # max函数返回最大值和索引的元组，我们仅需用到索引值作为标签
                correct += (predicted == labels).sum().item()
                total += predicted.size(0)
                total_loss += float(loss.item())
            now = time.time()

            self.writer.add_scalar('loss', total_loss, epoch+1)
            self.writer.add_scalar('accuracy', correct / total, epoch+1)
            print('epoch ' + str(epoch) + ': Accuracy on train dataset '
                  + str(correct / total) + ' total loss: ' + str(total_loss))
            print('lasts:' + str(now - start) + ' s')
        self.writer.close()

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
