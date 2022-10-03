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
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['imagenet', 'cifar10'], default='imagenet')
parser.add_argument('--path', help='The path of dataset', type=str, default='/datasets/imagenet_1k')
parser.add_argument('--pretrained', type=bool, default=True, help='Use the pretrained networks from Han Cai et. al')
parser.add_argument('--net', metavar='OFANET', choices=['ofa_mbv3_d234_e346_k357_w1.0', 'ofa_mbv3_d234_e346_k357_w1.2',
                                                        'ofa_proxyless_d234_e346_k357_w1.3', 'ofa_resnet50'],
                    default='ofa_resnet50', help='Select a available pretrained OFA network (all trained on imagenet)')
parser.add_argument('--net_typ', choices=['ResNet50', 'MobileNetV3'], help='Type of OFA network', default='ResNet50')
parser.add_argument('--net_path', help='Path to saved network model')  # only if not pretrained

parser.add_argument('--accuracy_predictor_path', help='Path to accuracy predictor model')
parser.add_argument('--num_layers', default=0, help='How many layers are used in the accuracy predictor')  # only for cifar10
parser.add_argument('--num_hidden_units', default=0, help='How many hidden units are used in the accuracy predictor')  # only for cifar10

args = parser.parse_args()

# TODO local definitions (need to be removed when using command line arguments)
args.dataset = 'imagenet'
args.path = 'C:\\Users\\bixi\\PycharmProjects\\OnceForAllFork\\datasets\\imagenet_1k'
args.pretrained = True
args.net = 'ofa_resnet50'
args.net_typ = 'ResNet50'
args.accuracy_predictor_path = r'C:\Users\bixi\PycharmProjects\OnceForAllFork\pretrained\ofa_nets\ofa_resnet50_acc_predictor.pth'

# how many random sub-networks should be tested
num_tests = 100

if args.dataset == 'cifar10':
    args.image_size = 32
    args.image_size_list = [32]
    args.num_classes = 10
elif args.dataset == 'imagenet':
    args.image_size = 224
    args.num_classes = 1000
else:
    raise NotImplementedError

if args.pretrained and args.dataset != 'imagenet':
    raise NotImplementedError('Pretrained networks are only available for ImageNet dataset')

args.batch_size = 100
args.n_workers = 0

if args.pretrained:
    ofa_network = ofa_net(args.net, pretrained=True)
    if 'resnet' in args.net:
        args.image_size_list = [128, 144, 160, 176, 192, 224, 240, 256]
    elif 'mbv3' in args.net:
        args.image_size_list = [128, 160, 192, 224]
    else:
        raise(NotImplementedError)
else:
    if args.net_typ == 'ResNet50':
        ofa_network = OFAResNets(
            n_classes=args.num_classes,
            dropout_rate=0,
            depth_list=[0, 1, 2],
            expand_ratio_list=[0.2, 0.25, 0.35],
            width_mult_list=[0.65, 0.8, 1.0],
            small_input_stem=True if args.dataset == 'cifar10' else False,
            dataset=args.dataset,
        )
        args.image_size_list = [128, 144, 160, 176, 192, 224, 240, 256]
    elif args.net_typ == 'MobileNetV3':
        ofa_network = net = OFAMobileNetV3(
            ks_list=[3, 5, 7],
            expand_ratio_list=[3, 4, 6],
            depth_list=[2, 3, 4],
        )
        args.image_size_list = [128, 160, 192, 224]
    else:
        raise NotImplementedError
    init = torch.load(
        args.net_path,
        map_location='cpu')['state_dict']
    ofa_network.load_state_dict(init)

run_config = ImagenetRunConfig(test_batch_size=args.batch_size, n_worker=args.n_workers, dataset=args.dataset, data_path=args.path)
run_config.data_provider.assign_active_img_size(args.image_size)  # assign image size: 128, 132, ..., 224

if args.net_typ == 'ResNet50':
    arch_encoder = ResNetArchEncoder(
        image_size_list=args.image_size_list,
        depth_list=ofa_network.depth_list,
        expand_list=ofa_network.expand_ratio_list,
        width_mult_list=ofa_network.width_mult_list,
        small_input_stem=ofa_network.small_input_stem,
    )
    if args.dataset == 'cifar10':
        if args.num_layers == 0 or args.num_hidden_units == 0:
            raise ValueError('For a custom accuracy predictor the architecture has to be defined.'
                             'Values for num_layers and num_hidden_units are missing')
        accuracy_predictor = AccuracyPredictor(
            arch_encoder=arch_encoder,
            n_layers=1,
            hidden_size=100,
            checkpoint_path=args.accuracy_predictor_path,
            device='cpu'
        )
    elif args.dataset == 'imagenet':
        accuracy_predictor = AccuracyPredictor(
            arch_encoder=arch_encoder,
            checkpoint_path=args.accuracy_predictor_path,
            device='cpu'
        )
    else:
        raise NotImplementedError
    # accuracy_predictor = torch.nn.DataParallel(accuracy_predictor)
elif args.net_typ == 'MobileNetV3':
    # TODO
    raise NotImplementedError
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
run_manager = RunManager('.tmp/eval_subnet', ofa_network, run_config, init=False)
for i in range(num_tests):

    net_config = ofa_network.sample_active_subnet()

    subnet = ofa_network.get_active_subnet(preserve_weight=True)
    net_info = get_net_info(subnet, [3, args.image_size, args.image_size], print_info=False)
    run_manager.reset_running_statistics(net=subnet)

    loss, (top1, top5) = run_manager.validate(net=subnet, is_test=True)
    net_config['image_size'] = random.choice(args.image_size_list)
    estimated_top1 = accuracy_predictor.predict_acc([net_config])
    # print('Results subnet: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (loss, top1, top5))

    net_info['loss'] = loss
    net_info['top1'] = top1  # TODO more digits after comma - increase batch size
    net_info['top5'] = top5
    net_info['net_config'] = net_config
    net_info['estimated_top1'] = estimated_top1.item()

    writer.writerow(net_info)
    file.flush()

