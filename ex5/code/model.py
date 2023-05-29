from torch import nn


class Generator(nn.Module):

    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.struct = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.output_size),
        )

    def forward(self, x):
        output = self.struct(x)
        return output


class Discriminator(nn.Module):

    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.struct = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.struct(x)
        return output


class DiscriminatorForWGAN(nn.Module):

    def __init__(self, input_size, output_size):
        super(DiscriminatorForWGAN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.struct = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.output_size),
            # 较GAN的判别器少了输出在1之内的限制，所以不再需要sigmoid函数的映射
        )

    def forward(self, x):
        output = self.struct(x)
        return output
