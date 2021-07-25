import argparse

import torch
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.imagenet_classification.elastic_nn.networks import OFAResNets
from ofa.nas.accuracy_predictor import *

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, default='imagenet',
    choices=[
        'imagenet', 'cifar10'
    ]
)
parser.add_argument(
    '--data_path', type=str, default=None, help='Path to dataset'
)
parser.add_argument(
    '--net', type=str, default='ResNet50',
    choices=[
        'ResNet50',
    ]
)
parser.add_argument('--net_path', default='', help='Path where network is stored')
parser.add_argument(
    '--experiment_id', type=str, default='', help='Id to identify the experiment'
)
args = parser.parse_args()

if args.dataset == 'cifar10':
    args.image_size = 32
    args.image_size_list = [32]
    args.num_classes = 10
    args.acc_dataset_size = 1000
    args.small_input_stem = True
else:
    args.image_size = 224
    args.image_size_list = [128, 160, 192, 224]
    args.num_classes = 1000
    args.acc_dataset_size = 16000
    args.small_input_stem = False

args.acc_dataset_folder = 'exp/exp_OFA' + args.net + '_' + args.dataset + '_' + args.experiment_id + '/acc_dataset'

batch_size = 100
n_worker = 0

ofa_network = OFAResNets(
    n_classes=args.num_classes,
    dropout_rate=0,
    depth_list=[0, 1, 2],
    expand_ratio_list=[0.2, 0.25, 0.35],
    width_mult_list=[0.65, 0.8, 1.0],
    small_input_stem=args.small_input_stem,
    dataset=args.dataset
)

init = torch.load(args.net_path, map_location='cpu')['state_dict']
ofa_network.load_state_dict(init)
ofa_network.cuda()

run_config = ImagenetRunConfig(test_batch_size=batch_size, n_worker=n_worker, dataset=args.dataset, data_path=args.data_path)
run_manager = RunManager('.tmp/eval_subnet', ofa_network, run_config, init=False)

accuracy_dataset = AccuracyDataset(args.acc_dataset_folder)
accuracy_dataset.build_acc_dataset(run_manager, ofa_network, args.acc_dataset_size, args.image_size_list)

