from torchviz import make_dot
import torch
from ofa.imagenet_classification.networks import *
from ofa.imagenet_classification.elastic_nn.networks import *


def drawModel(net, save_path, dataset='cifar10'):
    img_size = 32 if dataset == 'cifar10' else 224
    make_dot(net(torch.rand(1, 3, img_size, img_size))).render(save_path)


if __name__ == "__main__":
    dataset = 'cifar10'

    net = OFAResNets(
        n_classes=10,
        width_mult_list=1.0,
        expand_ratio_list=0.25,
        depth_list=[1, 2, 2, 1],
        dataset=dataset,
    )
    # net = ResNet50(n_classes=10, dataset=dataset)

    drawModel(net, 'logs/ofaResNet50', dataset=dataset)
