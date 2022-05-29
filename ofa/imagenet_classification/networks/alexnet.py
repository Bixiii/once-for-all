import torch.nn as nn

from ofa.utils.layers import set_layer_from_config, ConvLayer, IdentityLayer, LinearLayer
from ofa.utils.layers import ResNetBottleneckBlock, ResidualBlock
from ofa.utils import make_divisible, MyNetwork, MyGlobalAvgPool2d

__all__ = ['AlexNet']


class AlexNet(MyNetwork):
    """
    Modified to fit dataset size, @see https://github.com/soapisnotfat/pytorch-cifar10/blob/master/models/AlexNet.py
    """
    BASE_DEPTH_LIST = [1, 1, 1, 1]  # depth list [0, 1] ... +0 or +1 layer
    STAGE_WIDTH_LIST = [64, 192, 384, 256]
    ks_list = [3, 5]
    depth_list = [0, 1]

    def __init__(self, n_classes=10, bn_param=(0.1, 1e-5), dropout_rate=0, depth_list=1, ks_list=3):
        super(AlexNet, self).__init__()

        kernel_size = 5
        padding = 2
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=kernel_size, stride=2, padding=padding),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 192, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(192, 384, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x
