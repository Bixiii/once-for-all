import argparse
import time
import torch
import random
import numpy as np
import csv
import os
import pickle

from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from ofa.nas.efficiency_predictor import AnnetteLatencyModel
from ofa.tutorial import AccuracyPredictor, LatencyTable, EvolutionFinder, FLOPsTable
from utils import architecture_config_2_str
from visualize_subnet import Visualisation
from utils import logger, dict_2_str

"""
NAS for MobileNetV3 and Imagenet using pretrained networks from OnceForAll publication
"""

# dnndk = Xlinx ZCU 10, ncs2 = intel neural compute stick
annette_models = ['dnndk-mixed',
                  'dnndk-ref_roofline',
                  'dnndk-roofline',
                  'dnndk-statistical',
                  'ncs2-mixed',
                  'ncs2-ref_roofline',
                  'ncs2-roofline',
                  'ncs2-statistical',
                  ]

parser = argparse.ArgumentParser()
parser.add_argument('--latency', type=float, default=10.0, help='Latency constrain')
parser.add_argument('--constrain_type', type=str, default='flops', choices=['annette', 'flops'],
                    help='Mechanism used for latency estimation')
parser.add_argument('--annette_model', type=str, default=None, choices=annette_models,
                    help='Select which model should be used for ANNETTE latency estimation')
args = parser.parse_args()

latency_constraint = args.latency
constraint_type = args.constrain_type
annette_model = args.annette_model
parent_folder = 'results/'

# set random seed
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# set device
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)

# results to csv
results_file_name = parent_folder + 'mbv3_optimized_subnets.csv'
write_header = False if os.path.exists(results_file_name) else True
output_file = open(results_file_name, 'a+', newline='')
csv_data_fields = ['constrain_type', 'target_hardware', 'latency_constraint', 'estimated_latency', 'calculation_time',
                   'top1_acc',
                   'net_config']
csv_writer = csv.DictWriter(output_file, fieldnames=csv_data_fields)
if write_header:
    csv_writer.writeheader()

# accuracy predictor (pretrained from OFA repo)
accuracy_predictor = AccuracyPredictor(
    pretrained=True,
    device='cuda:0' if cuda_available else 'cpu'
)
# print(accuracy_predictor.model)

# OFA network
ks_list = [3, 5, 7]
depth_list = [2, 3, 4]
expand_list = [3, 4, 6]
num_classes = 1000

ofa_network = OFAMobileNetV3(
    n_classes=num_classes,
    dropout_rate=0,
    ks_list=ks_list,
    depth_list=depth_list,
    expand_ratio_list=expand_list,
)

# efficiency predictor
if constraint_type == 'flops':
    efficiency_predictor = FLOPsTable(
        device='cuda:0' if cuda_available else 'cpu',
        batch_size=1,
    )
elif constraint_type == 'annette':
    efficiency_predictor = AnnetteLatencyModel(ofa_network, model=annette_model)
else:
    raise NotImplementedError

# parameters for evolutionary algorithm
P = 100  # The size of population in each generation
N = 500  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    'constraint_type': constraint_type,  # Let's do FLOPs-constrained search
    'efficiency_constraint': latency_constraint,
    'mutate_prob': 0.1,  # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5,  # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': efficiency_predictor,  # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor,  # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
}

# build the evolution finder
logger.info('Create Evolution Finder')
logger.info('Parameters for NAS:\n%s' % dict_2_str(params))
finder = EvolutionFinder(**params)

# find optimal subnet
start = time.time()
logger.info('Start evolution search')
best_valids, best_info = finder.run_evolution_search()
logger.info('Finished evolution search')
end = time.time()

predicted_accuracy, subnet_config, estimated_latency = best_info
info_string = ('\n*****'
               '\nBest architecture for latency constrain <= %.2f ms'
               '\nIt achieves %.2f%s predicted accuracy with %.2f ms latency.'
               '\nSubnet configuration: \n%s'
               '\nNAS took %.2f seconds.'
               '\n*****' %
               (latency_constraint, predicted_accuracy * 100, '%', estimated_latency, str(subnet_config), end - start))
logger.info(info_string)
print(info_string)

# result to csv file
csv_writer.writerow(
    {'constrain_type': constraint_type,
     'target_hardware': annette_model,
     'latency_constraint': latency_constraint,
     'estimated_latency': estimated_latency,
     'calculation_time': end - start,
     'top1_acc': predicted_accuracy,
     'net_config': str(subnet_config)
     })
output_file.flush()

# visualize subnet config
drawing = Visualisation()
fig = drawing.mbv3_barchart(subnet_config,
                            save_path=parent_folder + (architecture_config_2_str(subnet_config) + '.png'),
                            title=(('Latency Constrain: %dms \nAccuracy: %.2f' % (
                            latency_constraint, predicted_accuracy * 100)) + '%'))
pickle.dump(fig, open(parent_folder + architecture_config_2_str(subnet_config) + '.pickle', 'wb'))
