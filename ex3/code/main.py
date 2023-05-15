from train_and_test.train_and_test import Train
import torchvision.transforms as transforms

if __name__ == '__main__':
    epoch = 90
    batch_size = 256
    size = 224
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    )
    train_adam = Train(epoch, )