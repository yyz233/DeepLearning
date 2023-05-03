from train_and_test.train import Train
from model.model import Model1, Model2
import torch

if __name__ == '__main__':
    # torch.set_num_threads(8)
    input_size = 28 * 28  # MNIST数据集中的每张图片由28x28个像素点构成
    hidden_size = 500  # 隐藏层大小
    num_classes = 10  # 手写数字识别，总共有10个类别的数字
    epochs = 30
    batch_size = 100  # 每一个batch的大小
    learning_rate = 0.0005  # 学习率
    train1 = Train(Model1, input_size, hidden_size, num_classes, batch_size, learning_rate, epochs)
    train1.train()
    train1.save()
    train1.test()
    train2 = Train(Model2, input_size, hidden_size, num_classes, batch_size, learning_rate, epochs)
    train2.train()
    train2.save()
    train2.test()