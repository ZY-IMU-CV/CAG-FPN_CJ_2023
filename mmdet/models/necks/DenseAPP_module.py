import torch
import torch.nn as nn
class DenseAPP(nn.Module):
    def __init__(self, num_channels=2048):
        super(DenseAPP, self).__init__()
        self.drop_out = 0.1
        self.channels1 = 512
        self.channels2 = 256
        self.num_channels = num_channels
        self.aspp3 = DenseBlock(self.num_channels, num1=self.channels1, num2=self.channels2, rate=3,
                                drop_out=self.drop_out)
        self.aspp6 = DenseBlock(self.num_channels + self.channels2 * 1, num1=self.channels1, num2=self.channels2,
                                rate=6,
                                drop_out=self.drop_out)
        self.aspp12 = DenseBlock(self.num_channels + self.channels2 * 2, num1=self.channels1, num2=self.channels2,
                                 rate=12,
                                 drop_out=self.drop_out)
        self.aspp18 = DenseBlock(self.num_channels + self.channels2 * 3, num1=self.channels1, num2=self.channels2,
                                 rate=18,
                                 drop_out=self.drop_out)
        self.aspp24 = DenseBlock(self.num_channels + self.channels2 * 4, num1=self.channels1, num2=self.channels2,
                                 rate=24,
                                 drop_out=self.drop_out)
        self.conv1x1 = nn.Conv2d(in_channels=5*self.channels2, out_channels=256, kernel_size=1)
        self.ConvGN = nn.GroupNorm(num_groups=32, num_channels=256)

    def forward(self, feature):
        aspp3 = self.aspp3(feature)
        feature = torch.cat((aspp3, feature), dim=1)
        aspp6 = self.aspp6(feature)
        feature = torch.cat((aspp6, feature), dim=1)
        aspp12 = self.aspp12(feature)
        feature = torch.cat((aspp12, feature), dim=1)
        aspp18 = self.aspp18(feature)
        feature = torch.cat((aspp18, feature), dim=1)
        aspp24 = self.aspp24(feature)

        x = torch.cat((aspp3, aspp6, aspp12, aspp18, aspp24), dim=1)
        out = self.ConvGN(self.conv1x1(x))
        return out

class DenseBlock(nn.Module):
    def __init__(self, input_num, num1, num2, rate, drop_out):
        super(DenseBlock, self).__init__()

        # C: 2048 --> 512 --> 256
        self.conv1x1 = nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)
        self.ConvGN = nn.GroupNorm(num_groups=32, num_channels=num1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dilaconv = nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3, padding=1 * rate, dilation=rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.ConvGN(self.conv1x1(x))
        x = self.relu1(x)
        x = self.dilaconv(x)
        x = self.relu2(x)
        x = self.drop(x)
        return x