import torchvision.models.alexnet
from train_and_test.train import Train

if __name__ == '__main__':
    epoch = 70
    batch_size = 1024
    train = Train(epoch, batch_size)
    train.train()
    train.save()
    train.test()