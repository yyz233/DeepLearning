from train_and_test.train_and_test import Train
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models import vgg11

if __name__ == '__main__':
    epoch = 2
    batch_size = 8
    size = 224
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    )
    # 使用torchvision自带VGG模型进行测试
    train_adam = Train(
        epoch,
        batch_size,
        size,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        scheduler=MultiStepLR,
        model=vgg11,
        transform=transform
    )
    # train_adam.train()
    train_adam.generate_test_csv()
    train_adam.save()