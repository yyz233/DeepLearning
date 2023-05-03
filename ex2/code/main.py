import torchvision.models.alexnet
from train_and_test.train import Train

if __name__ == '__main__':
    epoch = 90
    batch_size = 512
    train = Train(epoch, batch_size)
    train.train()
    train.save()
    train.test()