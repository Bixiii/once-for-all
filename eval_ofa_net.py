# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import torch
import argparse

from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.imagenet_classification.elastic_nn.networks import OFAResNets, OFAMobileNetV3
from ofa.model_zoo import ofa_net
from ofa.utils import get_net_info
import csv
import torch.onnx
from utils import export_as_onnx

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_path',
    help='The path of dataset',
    type=str,
    default='./datasets')
parser.add_argument(
    '--gpu',
    help='The gpu(s) to use',
    type=str,
    default='all')
parser.add_argument(
    '--batch-size',
    help='The batch on every device for validation',
    type=int,
    default=100)
parser.add_argument(
    '--workers',
    help='Number of workers',
    type=int,
    default=0)
parser.add_argument(
    '--pretrained_net',
    metavar='OFANET',
    default='',
    choices=['ofa_mbv3_d234_e346_k357_w1.0', 'ofa_mbv3_d234_e346_k357_w1.2', 'ofa_proxyless_d234_e346_k357_w1.3',
             'ofa_resnet50'],
    help='OFA networks')
parser.add_argument('--net', default='ResNet50', choices=['ResNet50', 'MobileNetV3'], help='Type of OFA network')
parser.add_argument('--net_path', default='', help='Path to saved network model')
parser.add_argument('--dataset', default='cifar10', choices=['imagenet', 'cifar10'])
parser.add_argument('--export_onnx', default=False, help='Export model as ONNX')
args = parser.parse_args()

output_file = './logs/ofa_evaluation_results.csv'

if args.dataset == 'cifar10':
    args.image_size = 32
    args.num_classes = 10
elif args.dataset == 'imagenet':
    args.image_size = 224
    args.num_classes = 1000
else:
    raise NotImplementedError

if args.gpu == 'all':
    device_list = range(torch.cuda.device_count())
    args.gpu = ','.join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(',')]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.batch_size = args.batch_size * max(len(device_list), 1)
ImagenetDataProvider.DEFAULT_PATH = args.data_path

# select and load a network
if args.pretrained_net:
    ofa_network = ofa_net(args.pretrained_net, pretrained=True)
else:
    if args.net == 'ResNet50':
        ofa_network = OFAResNets(
            n_classes=args.num_classes,
            dropout_rate=0,
            depth_list=[0, 1, 2],
            expand_ratio_list=[0.2, 0.25, 0.35],
            width_mult_list=[0.65, 0.8, 1.0],
            small_input_stem=True,
            dataset=args.dataset,
        )
    elif args.net == 'MobileNetV3':
        ofa_network = net = OFAMobileNetV3(
            ks_list=[3, 5, 7],
            expand_ratio_list=[3, 4, 6],
            depth_list=[2, 3, 4],
        )
    else:
        raise NotImplementedError
    init = torch.load(
        args.net_path,
        map_location='cpu')['state_dict']
    ofa_network.load_state_dict(init)

run_config = ImagenetRunConfig(test_batch_size=args.batch_size, n_worker=args.workers, dataset=args.dataset)
run_config.data_provider.assign_active_img_size(args.image_size)  # assign image size: 128, 132, ..., 224

csv_data_fields = ['params', 'flops', 'loss', 'top1', 'top5', 'net_config']
csv_writer = csv.DictWriter(open(output_file, 'w', newline=''), fieldnames=csv_data_fields)
csv_writer.writeheader()

net_config_list = []
# net_config_list.append({'d': [2, 2, 2, 2, 2], 'e': [0.25, 0.25, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.25, 0.35, 0.25, 0.25, 0.35, 0.25, 0.35], 'w': [1, 2, 2, 2, 2], 'image_size': 32})  # 600 - 95.72
# net_config_list.append({'d': [2, 2, 2, 1, 0], 'e': [0.25, 0.25, 0.2, 0.35, 0.35, 0.2, 0.25, 0.25, 0.35, 0.25, 0.25, 0.25, 0.2, 0.2, 0.25, 0.2, 0.2, 0.2], 'w': [1, 2, 0, 2, 2], 'image_size': 32})  # 300 - 95.65
# net_config_list.append({'d': [2, 0, 1, 1, 0], 'e': [0.2, 0.2, 0.25, 0.2, 0.2, 0.2, 0.2, 0.25, 0.35, 0.25, 0.25, 0.25, 0.2, 0.25, 0.2, 0.2, 0.2, 0.2], 'w': [2, 2, 0, 1, 0], 'image_size': 32})  # 150 - 95.28

""" Randomly sample a sub-network, 
    you can also manually set the sub-network using: 
        ofa_network.set_active_subnet(ks=7, e=6, d=4) 
"""
if not len(net_config_list):
    net_config_list.append(ofa_network.sample_active_subnet())
    print('Random subnet sampled ...')

for i, net_config in enumerate(net_config_list):
    ofa_network.set_active_subnet(**net_config)
    subnet = ofa_network.get_active_subnet(preserve_weight=True)
    net_info = get_net_info(subnet, [3, args.image_size, args.image_size], print_info=False)
    run_manager = RunManager('.tmp/eval_subnet', subnet, run_config, init=False)
    run_manager.reset_running_statistics(net=subnet)

    loss, (top1, top5) = run_manager.validate(net=subnet)
    net_info['loss'] = loss
    net_info['top1'] = top1
    net_info['top5'] = top5
    net_info['net_config'] = net_config

    if args.export_onnx:
        export_as_onnx(subnet.cpu(), 'accuracy_' + str(net_info['top1']) + '_OFAResNet50_' + str(i) + '.onnx')

    csv_writer.writerow(net_info)

    print('\n\t\tOFA subnet\n------------------------------')
    for key in net_info:
        print('{:<10} | {:<20}'.format(key, str(net_info[key])))
