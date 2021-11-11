"""
AlexNet
"""
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AlexNet(nn.Module):
    def __init__(self, class_num=5, init_weights=False):
        """
        Input (3, 224, 224)
        """
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),  # -> (48, 55, 55)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # -> (48, 27, 27)
            nn.Conv2d(48, 128, kernel_size=(5, 5), padding=(2, 2)),  # -> (128, 27, 27)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # -> (128, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=(3, 3), padding=(1, 1)),  # -> (192, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(3, 3), padding=(1, 1)),  # -> (192, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=(3, 3), padding=(1, 1)),  # -> (128, 13, 13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # -> (128, 6, 6)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, class_num)
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    my_net = AlexNet().to(device)
    print("my_net:\n", my_net)

    print("parameters:")
    for param in my_net.parameters():
        print("\tshape: {}".format(param.shape))

    print("named_parameters:")
    for name, param in my_net.named_parameters():
        print("\tname: {:<15}, shape: {}".format(name, param.shape))
