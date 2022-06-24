import torch
from utils import count_flops_thop, architecture_config_2_str
from ofa.imagenet_classification.elastic_nn.networks import OFAResNets
# from ofa.imagenet_classification.networks.resnets import *

# define parameters for OFA-network ResNet50
depth_list = [0, 1, 2]
expand_list = [0.2, 0.25, 0.35]
width_mult_list = [0.65, 0.8, 1.0]
small_input_stem = False

# crate OFA-network ResNet50
ofa_network = OFAResNets(
    n_classes=1000,
    dropout_rate=0,
    depth_list=depth_list,
    expand_ratio_list=expand_list,
    width_mult_list=width_mult_list,
    small_input_stem=small_input_stem,
    dataset='imagenet'
)

# subnet configurations for quick testing
subnet_config_small = {'d': [0, 0, 2, 2, 2], 'e': [0.35, 0.25, 0.35, 0.25, 0.35, 0.2, 0.2, 0.25, 0.2, 0.2, 0.35, 0.2, 0.25, 0.2, 0.25, 0.35, 0.2, 0.2], 'w': [0, 1, 2, 0, 0, 0], 'image_size': 128}
subnet_config_large = {'d': [2, 1, 2, 1, 1], 'e': [0.25, 0.35, 0.35, 0.25, 0.35, 0.35, 0.2, 0.2, 0.35, 0.25, 0.35, 0.35, 0.2, 0.25, 0.25, 0.35, 0.2, 0.25], 'w': [1, 0, 0, 2, 0, 2], 'image_size': 256}

subnet_config = subnet_config_small

# get current subnet-config
print('Extract subnet with config ' + str(subnet_config))
ofa_network.set_active_subnet(d=subnet_config['d'], e=subnet_config['e'], w=subnet_config['w'])
subnet = ofa_network.get_active_subnet()

# define shape of the input tensor
input_size = (1, 3, subnet_config['image_size'], subnet_config['image_size'])

# count the flops of the subnetwork - to show that the forward pass works
flops = count_flops_thop(subnet, input_size)
print(' -> MFLOPs: ' + str(flops))

# TODO: here you can use the instance of the subnetwork
x = torch.randn(*input_size, requires_grad=True).cpu()
subnet.forward(x)
