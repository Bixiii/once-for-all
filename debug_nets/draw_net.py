from torchviz import make_dot
import torch
from ofa.imagenet_classification.networks import *



def drawModel(net, save_path):
    make_dot(net(torch.rand(1, 3, 32, 32))).render(save_path)


if __name__ == "__main__":
    net = ResNet50(n_classes=10, dataset='cifar10')
    drawModel(net, 'logs/ofaResNet50')
