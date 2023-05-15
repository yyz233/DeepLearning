import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from collections import defaultdict
import cv2
import csv
import os

# 全局标签名-数值的映射，由于它只会在train_data_read函数之中被初始化，所以一定要先执行train_data_read函数
label_map = defaultdict(int)


class MyDataset(Dataset):

    def __init__(self, data, label, transform):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.transform(self.data[item][:]), self.label[item]


def train_data_read(size):
    """
    :param size: 将图像resize为size*size
    :return: data 和 label
    """
    path = ".\\data\\train"
    image_dir_paths = next(os.walk(path))[1]  # 获取所有子目录路径
    data = []
    labels = []
    # 读取数据
    for i, image_dir_path in enumerate(image_dir_paths):
        label_map[image_dir_path] = i
        image_names = next(os.walk(path + '\\' + image_dir_path))[2]
        for image_name in image_names:
            image = cv2.imread(path + '\\' + image_dir_path + '\\' + image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            data.append(image)
            labels.append(i)
    for i in label_map:
        print(i, label_map[i])

    return data, labels


def test_data_read(size):
    """
    :param size: 将图像resize为size*size
    :return: data 和 label
    """
    path = '.\\data\\test'
    data = []
    labels = []
    image_label_map = defaultdict(str)
    with open('.\\data\\sample_submission.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        next(spamreader)
        for row in spamreader:
            image_label_map[row[0]] = row[1]

    images_names = next(os.walk(path))[2]
    for image_name in images_names:
        image = cv2.imread(path + '\\' + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        data.append(image)
        labels.append(label_map[image_label_map[image_name]])
    return data, labels


def data_process(batch_size, size):
    """
    如果要读取数据，只需要调用这个函数即可
    :param batch_size: 含义就像他的名字一样
    :param size: 图像resize之后的大小为size*size
    :return: 两个torch.utils.data.DataLoader对象的实例，分别意味着train_data_loader 和test_data_loader
    注：transform部分我不太明白什么样的比较好，于是我采用了之前实验所使用的学长的transform ————yyz
    """
    my_transform = transforms.Compose(
        [transforms.ToPILImage(),
         # transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    )
    train_data, train_labels = train_data_read(size)
    test_data, test_labels = test_data_read(size)
    train_dataset = MyDataset(train_data, train_labels, my_transform)
    test_dataset = MyDataset(test_data, test_labels, my_transform)
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
    )
