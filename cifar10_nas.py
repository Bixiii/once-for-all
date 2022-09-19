import os
import time
import argparse
import csv

from ofa.imagenet_classification.elastic_nn.networks import OFAResNets, OFAMobileNetV3

from ofa.nas.efficiency_predictor import ResNet50LatencyTable, ResNet50FLOPsModel, Mbv3FLOPsModel

from ofa.nas.accuracy_predictor import *

from ofa.nas.search_algorithm import EvolutionFinder
from utils import export_as_onnx

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='ResNet50', choices=['ResNet50', 'MobileNetV3'])
parser.add_argument('--dataset', default='cifar10', choices=['cifar10'])
parser.add_argument('--accuracy_predictor_path', default=None, help='Path to the trained accuracy predictor')
args = parser.parse_args()

# TODO local definitions (need to be removed when using command line arguments)
args.accuracy_predictor_path = r'C:\Users\bixi\PycharmProjects\OnceForAllFork\exp\exp_OFAResNet50_cifar10_exp-0.0.2\accuracy_predictor\acc_predictor_model_best.pth.tar'
args.net = 'MobileNetV3'
args.dataset = 'cifar10'

if not args.accuracy_predictor_path:
    raise FileExistsError('You need to specify the path to the trained accuracy predictor model.')

# NAS constrains
target_hardware = 'flops'
latency_constraints = [500, 450, 400, 350, 300, 250, 200, 150]

# CSV
output_file_name = './logs/ofa_optimized_subnets.csv'
output_file = open(output_file_name, 'w', newline='')
csv_data_fields = ['target_hardware', 'latency_constraint', 'estimated_latency', 'calculation_time', 'top1_acc', 'net_config']
csv_writer = csv.DictWriter(output_file, fieldnames=csv_data_fields)
csv_writer.writeheader()

# Accuracy predictor
args.n_layers = 1
args.hidden_size = 100

# OFA network settings
if args.net == 'ResNet50':
    args.depth_list = [0, 1, 2]
    args.expand_list = [0.2, 0.25, 0.35]
    args.width_mult_list = [0.65, 0.8, 1.0]
    args.small_input_stem = True
elif args.net == 'MobileNetV3':
    args.ks_list = [3, 5, 7]
    args.depth_list = [2, 3, 4]
    args.expand_list = [3, 4, 6]
    if args.dataset == 'cifar10':
        raise NotImplementedError
else:
     raise NotImplementedError

# Dataset
if args.dataset == 'cifar10':
    args.image_size_list = [32]
    args.num_classes = 10
else:
    raise NotImplementedError

if args.net == 'ResNet50':
    arch_encoder = ResNetArchEncoder(
        image_size_list=args.image_size_list,
        depth_list=args.depth_list,
        expand_list=args.expand_list,
        width_mult_list=args.width_mult_list,
        small_input_stem=args.small_input_stem,
    )
elif args.net == 'MobileNetV3':
    arch_encoder = MobileNetArchEncoder(
        image_size_list=args.image_size_list,
        depth_list=args.depth_list,
        expand_list=args.expand_list,
    )
else:
    raise NotImplementedError

accuracy_predictor = AccuracyPredictor(
    arch_encoder=arch_encoder,
    n_layers=args.n_layers,
    hidden_size=args.hidden_size,
    # checkpoint_path=args.accuracy_predictor_path
)

if args.net == 'ResNet50':
    ofa_network = OFAResNets(
        n_classes=args.num_classes,
        dropout_rate=0,
        depth_list=args.depth_list,
        expand_ratio_list=args.expand_list,
        width_mult_list=args.width_mult_list,
        small_input_stem=args.small_input_stem,
        dataset=args.dataset
    )
    efficiency_predictor = ResNet50FLOPsModel(ofa_network)
elif args.net == 'MobileNetV3':
    ofa_network = OFAMobileNetV3(
        n_classes=args.num_classes,
        dropout_rate=0,
        depth_list=args.depth_list,
        expand_ratio_list=args.expand_list,
    )
    efficiency_predictor = Mbv3FLOPsModel(ofa_network)
else:
    raise NotImplementedError

# Hyper-parameters for the evolutionary search process
P = 10  # The size of population in each generation
N = 100  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation

for latency_constraint in latency_constraints:
    params = {
        'constraint_type': target_hardware,
        'efficiency_constraint': latency_constraint,
        'arch_mutate_prob': 0.1,
        'mutation_ratio': 0.5,
        'efficiency_predictor': efficiency_predictor,
        'accuracy_predictor': accuracy_predictor,
        'population_size': P,
        'max_time_budget': N,
        'parent_ratio': r,
    }

    finder = EvolutionFinder(**params)

    # start searching
    result_list = []
    st = time.time()
    print('Start search for best subnet architecture ...')
    best_valid, best_info = finder.run_evolution_search(latency_constraint)
    result_list.append(best_info)
    ed = time.time()
    print('Found best architecture on %s with latency <= %.2f ms in %.2f seconds! '
          'It achieves %.2f%s predicted accuracy with %.2f ms latency on %s.' 
          '\nArchitecture configuration: %s' %
          (target_hardware, latency_constraint, ed-st, best_info[0] * 100, '%', best_info[-1], target_hardware,
           str(best_info[1])))

    csv_writer.writerow(
        {'target_hardware': target_hardware,
         'latency_constraint': latency_constraint,
         'estimated_latency': best_info[-1],
         'calculation_time': ed-st,
         'top1_acc': best_info[0],
         'net_config': best_info[1]
         })
    output_file.flush()

    best_architecture = best_info[1]
    best_architecture.pop('image_size')
    ofa_network.set_active_subnet(**best_architecture)
    best_subnetwork = ofa_network.get_active_subnet()  # TODO for some reason this does not work with MobileNetV3

    os.makedirs('exp/optimized_subnets', exist_ok=True)
    subnet_name = 'exp/optimized_subnets/pt_' + 'OFA' + args.net + '_' + str(min(args.image_size_list)) + 'x' +\
                  str(min(args.image_size_list)) + '_' + args.dataset + '_' + target_hardware +\
                  '-latency' + str(latency_constraint) + '-acc' + str('0{:.0f}'.format(best_info[0]*10000)) + '.onnx'
    export_as_onnx(best_subnetwork, file_name=subnet_name)
    print('Saved optimized subnet as ' + subnet_name)


