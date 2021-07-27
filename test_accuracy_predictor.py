# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import torch
import argparse

from ofa.nas.accuracy_predictor import ResNetArchEncoder, AccuracyPredictor

from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.imagenet_classification.elastic_nn.networks import OFAResNets, OFAMobileNetV3
from ofa.model_zoo import ofa_net
from ofa.utils import get_net_info
import csv
import torch.onnx

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['imagenet', 'cifar10'])
parser.add_argument(
    '-p',
    '--path',
    help='The path of dataset',
    type=str,
    default='/dataset/imagenet')

parser.add_argument('--pretrained', type=bool, default=False, help='Use the pretrained networks from Han Cai et. al')
parser.add_argument(
    '-n',
    '--net',
    metavar='OFANET',
    default='ofa_resnet50',
    choices=['ofa_mbv3_d234_e346_k357_w1.0', 'ofa_mbv3_d234_e346_k357_w1.2', 'ofa_proxyless_d234_e346_k357_w1.3',
             'ofa_resnet50'],
    help='Select a available pretrained OFA network (all trained on imagenet)')

parser.add_argument('--net_typ', choices=['ResNet50', 'MobileNetV3'], help='Type of OFA network')
parser.add_argument('--net_path', help='Path to saved network model')

parser.add_argument('--num_layers', default=2, help='How many layers are used in the accuracy predictor')
parser.add_argument('--num_hidden_units', default=150, help='How many hidden units are used in the accuracy predictor')
parser.add_argument('--accuracy_predictor_path', help='Path to accuracy predictor model')

args = parser.parse_args()

num_tests = 3

if args.dataset == 'cifar10':
    args.image_size = 32
    args.image_size_list = [32]
    args.num_classes = 10
elif args.dataset == 'imagenet':
    args.image_size = 224
    args.image_size_list = [128, 160, 192, 224]
    args.num_classes = 1000
else:
    raise NotImplementedError

if args.pretrained and args.dataset != 'imagenet':
    raise NotImplementedError('Pretrained networks are only available for Imagnet dataset')

args.batch_size = 100
args.n_workers = 0

if args.pretrained:
    ofa_network = ofa_net(args.net, pretrained=True)
else:
    if args.net_typ == 'ResNet50':
        ofa_network = OFAResNets(
            n_classes=args.num_classes,
            dropout_rate=0,
            depth_list=[0, 1, 2],
            expand_ratio_list=[0.2, 0.25, 0.35],
            width_mult_list=[0.65, 0.8, 1.0],
            small_input_stem=True,
            dataset=args.dataset,
        )
    elif args.net_typ == 'MobileNetV3':
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

run_config = ImagenetRunConfig(test_batch_size=args.batch_size, n_worker=args.n_workers, dataset=args.dataset)
run_config.data_provider.assign_active_img_size(args.image_size)  # assign image size: 128, 132, ..., 224

if args.net_typ == 'ResNet50':
    arch_encoder = ResNetArchEncoder(
        image_size_list=args.image_size_list,
        depth_list=ofa_network.depth_list,
        expand_list=ofa_network.expand_ratio_list,
        width_mult_list=ofa_network.width_mult_list,
        small_input_stem=ofa_network.small_input_stem,
    )
    accuracy_predictor = AccuracyPredictor(
        arch_encoder=arch_encoder,
        n_layers=1,
        hidden_size=100,
        checkpoint_path=args.accuracy_predictor_path,
    )
    # accuracy_predictor = torch.nn.DataParallel(accuracy_predictor)
else:
    raise NotImplementedError


data_fields = ['params', 'flops', 'loss', 'top1', 'top5', 'net_config', 'estimated_top1']
file = open('eval_acc_pred_data_points.csv', 'w', newline='')
writer = csv.DictWriter(file, fieldnames=data_fields)
writer.writeheader()

""" Randomly sample a sub-network, 
    you can also manually set the sub-network using: 
        ofa_network.set_active_subnet(ks=7, e=6, d=4) 
"""
for i in range(num_tests):
    net_config = ofa_network.sample_active_subnet()

    """ Test sampled subnet 
    """
    subnet = ofa_network.get_active_subnet(preserve_weight=True)
    net_info = get_net_info(subnet, [3, args.image_size, args.image_size], print_info=False)
    run_manager = RunManager('.tmp/eval_subnet', subnet, run_config, init=False)
    run_manager.reset_running_statistics(net=subnet)

    loss, (top1, top5) = run_manager.validate(net=subnet)
    net_config['image_size'] = 32
    estimated_top1 = accuracy_predictor.predict_acc([net_config])
    # print('Results subnet: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (loss, top1, top5))

    net_info['loss'] = loss
    net_info['top1'] = top1
    net_info['top5'] = top5
    net_info['net_config'] = net_config
    net_info['estimated_top1'] = estimated_top1.item()

    writer.writerow(net_info)
    file.flush()

