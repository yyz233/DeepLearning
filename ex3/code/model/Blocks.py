from torch import nn
from torch.nn import functional as F

class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, img):
        identity = img
        out = self.conv1(img)
        out = self.conv2(out)
        out += identity
        return F.relu(out)


class BlockWithDownSample(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(BlockWithDownSample, self).__init__()
        self.downSample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1,
                      stride=stride, padding=0),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, img):
        identity = self.downSample(img)
        out = self.conv1(img)
        out = self.conv2(out)
        out += identity
        return F.relu(out)
