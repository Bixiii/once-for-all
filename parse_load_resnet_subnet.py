import torch
from ofa.imagenet_classification.elastic_nn.networks import OFAResNets
from result_net_configs import *
from utils import *
import csv
from ofa.imagenet_classification.networks.resnets import ResNets

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

# read configs for sub-networks from csv
csv_file_path = r"D:\_Data\University\Master Thesis\Data\ofa_ncs2\resnet_subnet_metrics\metrics_ofa_resnet_subnets_20220609_11-01-27.csv"
f = open(csv_file_path)
csv_reader = csv.reader(f)

# parse csv
csv_header = next(csv_reader)
info_subnets = []
for row in csv_reader:
    info_subnets.append({'file_id': row[-1], 'config': string_2_config_dict(row[0])})

# import
path = 'ofa_resnet.pth'
init = torch.load(path)
ofa_network.load_state_dict(init)

# create subnet for each config
for info_subnet in info_subnets[0:1]:

    # get current subnet
    subnet_config = info_subnet['config']
    ofa_network.set_active_subnet(d=subnet_config['d'], e=subnet_config['e'], w=subnet_config['w'])
    subnet = ofa_network.get_active_subnet()

    print('Metrics for file with id <' + info_subnet['file_id'] + '>')
    input_size = (1, 3, subnet_config['image_size'], subnet_config['image_size'])
    flops = count_flops_thop(subnet, input_size)
    print('flops: ' + str(flops))

    # x = torch.randn(*input_size, requires_grad=True).cpu()
    # subnet.forward(x)
