import torch.nn as nn


class Model1(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Model1, self).__init__()
        self.input2hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(p=0.3),  # 预防过拟合，采用p=0.3的概率不激活
            nn.LeakyReLU()  # 较ReLu更好
        )
        self.hidden2output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.input2hidden(x)
        x = self.hidden2output(x)
        return x

    # 方便打印结果
    def __str__(self):
        return "Model1"


class Model2(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Model2, self).__init__()
        self.input2hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(p=0.3),  # 预防过拟合，采用p=0.3的概率不激活
            nn.LeakyReLU()  # 较ReLu更好
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, 500),
            nn.Dropout(p=0.3),  # 预防过拟合，采用p=0.3的概率不激活
            nn.LeakyReLU()  # 较ReLu更好
        )
        self.hidden2output = nn.Linear(500, num_classes)

    def forward(self, x):
        x = self.input2hidden(x)
        x = self.layer2(x)
        x = self.hidden2output(x)
        return x

    # 方便打印结果
    def __str__(self):
        return "Model2"

