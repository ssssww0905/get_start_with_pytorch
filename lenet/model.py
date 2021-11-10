"""
LeNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # x = torch.flatten(x, start_dim=1)  # dim_0 : batch_size
        x = nn.Flatten()(x)  # first dim to flatten (default = 1).
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    my_net = LeNet().to(device)
    print("my_net:\n", my_net)

    print("parameters:")
    for param in my_net.parameters():
        print("\tshape: {}".format(param.shape))

    print("named_parameters:")
    for name, param in my_net.named_parameters():
        print("\tname: {:<15}, shape: {}".format(name, param.shape))

