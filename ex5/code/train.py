from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from image_generate import draw_scatter, draw_background
from torch.utils.data import DataLoader, Dataset
import torch.autograd as autograd

class MyDataset(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        return torch.tensor(np.array(self.data[item]).astype(float)),\
            torch.tensor(np.array(self.label[item]).astype(float))

    def __len__(self):
        return len(self.data)


def data_process(batch_size, input_size):
    mt = loadmat('./data/points.mat')
    data = mt['xx']
    num = data.shape[0]
    test_size = int(num/10)
    train_size = num - test_size
    # 进行打乱，使得训练集/测试集划分时，都是近似“M”的图形
    np.random.shuffle(data)
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    train_ = data[:train_size]
    test_ = data[train_size:]
    _train = torch.randn(train_size, input_size)
    _test = torch.randn(test_size, input_size)
    for i in range(train_size):
        train_data.append(_train[i])
        train_label.append(train_[i])
    for i in range(test_size):
        test_data.append(_test[i])
        test_label.append(test_[i])
    train_dataset = MyDataset(train_data, train_label)
    test_dataset = MyDataset(test_data, test_label)
    return test_, DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    ), DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )


def generate_process_pic(generator, discriminator, input_size, path, epoch, test_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 测试状态
    generator.eval()
    generator.zero_grad()
    discriminator.eval()
    discriminator.zero_grad()
    # 处理生成的数据
    input_data = torch.randn(1000, input_size).to(device)
    output = generator(input_data).to(device)
    background = draw_background(discriminator)
    output_cpu = output.cpu().detach().numpy()
    draw_scatter(test_data, 'b')
    draw_scatter(output_cpu, 'r')
    plt.savefig('./result/' + path + '/epoch' + str(epoch + 1))
    background.remove()
    plt.cla()
    # 回到训练过程
    generator.train()
    discriminator.train()


def train_GAN(epochs, train_dataloader, generator, discriminator,
              generator_optimizer, discriminator_optimizer, input_size, test_data, task):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.train()
    discriminator.train()
    for epoch in range(epochs):
        for it, (data, label) in enumerate(train_dataloader):
            # 训练过程
            # 前向传播
            data = torch.randn(data.shape[0], input_size)
            data = data.to(device)
            label = label.to(device)
            data = data.type_as(generator.struct[0].weight).to(device)
            label = label.type_as(generator.struct[0].weight).to(device)
            output = generator(data).to(device)
            # 判别器判别生成概率
            label_prob = discriminator(label).to(device)
            output_prob = discriminator(output).to(device)

            # loss计算
            discriminator_loss = - torch.mean(torch.log(label_prob) + torch.log(1. - output_prob)).to(device)
            generator_loss = torch.mean(torch.log(1. - output_prob)).to(device)
            # 更新判别器参数
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()
            # 更新生成器参数
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

        if epoch % 10 == 9:
            # 这里进行图像处理
            generate_process_pic(generator, discriminator, input_size, task, epoch, test_data)
            print('epoch ' + str(epoch) + ' finished!')


def train_WGAN(epochs, train_dataloader, generator, discriminator,
              generator_optimizer, discriminator_optimizer, input_size, test_data, clamp, task):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.train()
    discriminator.train()
    for epoch in range(epochs):
        for it, (data, label) in enumerate(train_dataloader):
            # 训练过程
            # 前向传播
            data = torch.randn(data.shape[0], input_size)
            data = data.to(device)
            label = label.to(device)
            data = data.type_as(generator.struct[0].weight).to(device)
            label = label.type_as(generator.struct[0].weight).to(device)
            output = generator(data).to(device)
            # 判别器判别生成概率
            label_prob = discriminator(label).to(device)
            output_prob = discriminator(output).to(device)
            # loss计算
            # 由于是WGAN所以这里的loss函数计算方式需要更改
            discriminator_loss = - torch.mean(output_prob - label_prob).to(device)
            generator_loss = torch.mean(output_prob).to(device)
            # 更新判别器参数
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()
            # 限制判别器参数大小
            for p in discriminator.parameters():
                p.data.clamp_(-clamp, clamp)
            # 更新生成器参数
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

        if epoch % 10 == 9:
            # 这里进行图像处理
            generate_process_pic(generator, discriminator, input_size, task, epoch, test_data)
            print('epoch ' + str(epoch) + ' finished!')


def compute_gradient_penalty(D, real_samples, fake_samples):
    Tensor = torch.cuda.FloatTensor
    alpha = torch.rand(real_samples.shape).to('cuda')
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = autograd.Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean()
    return gradient_penalty


def train_WGAN_GP(epochs, train_dataloader, generator, discriminator,
              generator_optimizer, discriminator_optimizer, input_size, test_data, clamp, task):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.train()
    discriminator.train()
    for epoch in range(epochs):
        for it, (data, label) in enumerate(train_dataloader):
            # 训练过程
            # 前向传播
            data = torch.randn(data.shape[0], input_size)
            data = data.to(device)
            label = label.to(device)
            data = data.type_as(generator.struct[0].weight).to(device)
            label = label.type_as(generator.struct[0].weight).to(device)
            output = generator(data).to(device)
            # 判别器判别生成概率
            label_prob = discriminator(label).to(device)
            output_prob = discriminator(output).to(device)
            # 相较于WGAN，WGAN-GP多了gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, label, output)
            # loss计算
            # 由于是WGAN GP所以这里的loss函数计算方式需要更改
            discriminator_loss = (- label_prob.mean() + output_prob.mean() + .001 * gradient_penalty).to(device)
            generator_loss = - torch.mean(output_prob).to(device)
            # 更新判别器参数
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()
            # 限制判别器参数大小
            for p in discriminator.parameters():
                p.data.clamp_(-clamp, clamp)
            # 更新生成器参数
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

        if epoch % 10 == 9:
            # 这里进行图像处理
            generate_process_pic(generator, discriminator, input_size, task, epoch, test_data)
            print('epoch ' + str(epoch) + ' finished!')
