import torch
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=c_in)

        self.conv2 = nn.Conv2d(c_out, self.expansion * c_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=c_out)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, c_in, c_out, stride, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)

        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)

        self.conv3 = nn.Conv2d(c_out, self.expansion * c_out, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * c_out)

        self.relu = nn.ReLU(inplace=True)
        self.dowmsample = downsample

    def forward(self, x):
        identity = x
        if self.dowmsample is not None:
            identity = self.dowmsample(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, block_list, class_num=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.c_in = 64  # iterate in _make_layer

        self.conv1 = nn.Conv2d(3, self.c_in, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(self.c_in)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = self._make_layer(block, 64, block_list[0], stride=(1, 1))
        self.conv3 = self._make_layer(block, 128, block_list[1], stride=(2, 2))
        self.conv4 = self._make_layer(block, 256, block_list[2], stride=(2, 2))
        self.conv5 = self._make_layer(block, 512, block_list[3], stride=(2, 2))

        if self.include_top:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(512 * block.expansion, class_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, c_out, num, stride):
        downsample = None
        if stride != (1, 1) or self.c_in != c_out * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.c_in, c_out * block.expansion, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(c_out * block.expansion)
            )
        block_list = [block(self.c_in, c_out, stride, downsample)]
        self.c_in = c_out * block.expansion
        for i in range(num - 1):
            block_list.append(block(self.c_in, c_out, (1, 1)))

        return nn.Sequential(*block_list)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        if self.include_top:
            x = self.avg_pool(x)
            x = self.flatten(x)
            x = self.fc(x)

        return x


def resnet34(class_num=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], class_num=class_num, include_top=include_top)


def resnet50(class_num=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], class_num=class_num, include_top=include_top)


if __name__ == "__main__":
    print("resnet34:\n{}".format(resnet34()))
    print("resnet50:\n{}".format(resnet50()))
