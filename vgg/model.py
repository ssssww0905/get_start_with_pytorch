import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'


vgg_name2model = {
    "vgg_11": [
        64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"
    ],
    "vgg_16": [
        64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"
    ],
}


class VGGNet(nn.Module):
    def __init__(self, model_name, class_num=1000, init_weights=False):
        """
        Input : (3, 224, 224)
        """
        super(VGGNet, self).__init__()
        assert model_name in vgg_name2model.keys(),\
            "not support {} model now!".format(model_name)
        model_list = vgg_name2model[model_name]
        features_seq = []
        c_in = 3
        for i in model_list:
            if isinstance(i, int):
                features_seq.append(nn.Conv2d(c_in, i, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
                features_seq.append(nn.ReLU(inplace=True))
                c_in = i
            elif isinstance(i, str) and i == "M":
                features_seq.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            else:
                raise ValueError("can not add {} layer!".format(i))
        self._features = nn.Sequential(*features_seq)
        self._flatten = nn.Flatten()
        self._classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_num),
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self._features(x)
        x = self._flatten(x)
        x = self._classifier(x)
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
    model = VGGNet("vgg_16")
    print("model:\n", model)

    print("named_parameters:")
    for name, param in model.named_parameters():
        print("\tname: {:<15}, shape: {}".format(name, param.shape))
