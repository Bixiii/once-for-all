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

parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--path',
    help='The path of dataset',
    type=str,
    default='/dataset/imagenet')
parser.add_argument(
    '-g',
    '--gpu',
    help='The gpu(s) to use',
    type=str,
    default='all')
parser.add_argument(
    '-b',
    '--batch-size',
    help='The batch on every device for validation',
    type=int,
    default=100)
parser.add_argument(
    '-j',
    '--workers',
    help='Number of workers',
    type=int,
    default=0)
parser.add_argument(
    '-n',
    '--net',
    metavar='OFANET',
    default='ofa_resnet50',
    choices=['ofa_mbv3_d234_e346_k357_w1.0', 'ofa_mbv3_d234_e346_k357_w1.2', 'ofa_proxyless_d234_e346_k357_w1.3',
             'ofa_resnet50'],
    help='OFA networks')
parser.add_argument('--pretrained', type=bool, default=False, help='Use the pretrained networks from Han Cai et. al')
parser.add_argument('--net_path', help='Path to saved network model')
parser.add_argument('--net_type', choices=['ResNet50', 'MobileNetV3'], help='Type of OFA network')
parser.add_argument('--dataset', choices=['imagenet', 'cifar10'])
args = parser.parse_args()

if args.dataset == 'cifar10':
    args.image_size = 32
    args.num_classes = 10
elif args.dataset == 'imagenet':
    args.image_size = 224
    args.num_classes = 1000
else:
    raise NotImplementedError


def export_as_onnx(net, file_name):
    x = torch.randn(1, 3, 32, 32, requires_grad=True).cpu()

    torch.onnx.export(
        net,
        x,
        'logs/' + file_name + '.onnx',
        export_params=True,
        )


if args.gpu == 'all':
    device_list = range(torch.cuda.device_count())
    args.gpu = ','.join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(',')]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.batch_size = args.batch_size * max(len(device_list), 1)
ImagenetDataProvider.DEFAULT_PATH = args.path

if args.pretrained:
    ofa_network = ofa_net(args.net, pretrained=True)
else:

    if args.net_type == 'ResNet50':
        ofa_network = OFAResNets(
            n_classes=args.num_classes,
            dropout_rate=0,
            depth_list=[0, 1, 2],
            expand_ratio_list=[0.2, 0.25, 0.35],
            width_mult_list=[0.65, 0.8, 1.0],
            small_input_stem=True,
            dataset=args.dataset,
        )
    elif args.net_type == 'MobileNetV3':
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

""" Randomly sample a sub-network, 
    you can also manually set the sub-network using: 
        ofa_network.set_active_subnet(ks=7, e=6, d=4) 
"""
evaluation_data = []
evaluation_data_fields = ['params', 'flops', 'loss', 'top1', 'top5', 'net_config']

for i in range(100):
    net_config = ofa_network.sample_active_subnet()
    # random_net_config = {'d': [2, 1, 0, 2, 2], 'e': [0.35, 0.35, 0.2, 0.25, 0.25, 0.35, 0.25, 0.35, 0.2, 0.35, 0.35, 0.25, 0.25, 0.35, 0.2, 0.25, 0.35, 0.2], 'w': [1, 2, 0, 0, 2]}
    # smallest_net_config = {'d': [0, 0, 0, 0, 0], 'e': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 'w': [0, 0, 0, 0, 0]}
    # largest_net_config = {'d': [2, 2, 2, 2, 2], 'e': [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35], 'w': [2, 2, 2, 2, 2]}
    # net_config = largest_net_config
    # ofa_network.set_active_subnet(**net_config)

    """ Test sampled subnet 
    """
    subnet = ofa_network.get_active_subnet(preserve_weight=True)
    net_info = get_net_info(subnet, [3, args.image_size, args.image_size], print_info=False)
    run_manager = RunManager('.tmp/eval_subnet', subnet, run_config, init=False)
    run_manager.reset_running_statistics(net=subnet)

    # print('Test random subnet:')
    # print(subnet.module_str)

    loss, (top1, top5) = run_manager.validate(net=subnet)
    # print('Results subnet: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (loss, top1, top5))

    net_info['loss'] = loss
    net_info['top1'] = top1
    net_info['top5'] = top5
    net_info['net_config'] = net_config

    evaluation_data.append(net_info)


writer = csv.DictWriter(open('eval_data_points.csv', 'w', newline=''), fieldnames=evaluation_data_fields)
writer.writeheader()
for data in evaluation_data:
    writer.writerow(data)
# print('Eval result: ', str(net_info))

# print('Export selected subnet as ONNX')
# export_as_onnx(subnet.cpu(), 'subnet.onnx')