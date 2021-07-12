from torchviz import make_dot
import torch
from ofa.imagenet_classification.networks import *
from ofa.imagenet_classification.elastic_nn.networks import *


def drawModel(net, save_path, img_size):
    make_dot(net(torch.rand(1, 3, img_size, img_size))).render(save_path)


if __name__ == "__main__":

    teacher_model = ResNet50(
        n_classes=1000,
        expand_ratio=0.35,
        depth_list=[4, 4, 6, 4]
    )
    drawModel(teacher_model, 'logs/teacher', img_size=224)

    ofa_model = OFAResNets(
        n_classes=1000,
        expand_ratio_list=0.35,
        small_input_stem=True
    )
    drawModel(ofa_model, 'logs/ofa', img_size=224)

    # net = OFAResNets(
    #     n_classes=10,
    #     width_mult_list=1.0,
    #     expand_ratio_list=0.25,
    #     depth_list=[1, 2, 2, 1],
    # )
    # drawModel(net, 'logs/ofaResNet50', img_size=224)
    #
    # net = ResNet50(n_classes=10)
    # drawModel(net, 'logs/ResNet', img_size=224)

