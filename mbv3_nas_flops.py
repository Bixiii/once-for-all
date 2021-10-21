import time
import torch
import random
import numpy as np
import csv
import os

from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from ofa.tutorial import AccuracyPredictor, LatencyTable, EvolutionFinder, FLOPsTable
from visualize_subnet import Visualisation

"""
NAS for MobileNetV3 and Imagenet using pretrained networks from OnceForAll publication
"""

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
results_file_name = 'mbv3_optimized_subnets.csv'
write_header = False if os.path.exists(results_file_name) else True
output_file = open(results_file_name, 'a+', newline='')
csv_data_fields = ['target_hardware', 'latency_constraint', 'estimated_latency', 'calculation_time', 'top1_acc',
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

# efficiency predictor - flop table
flops_lookup_table = FLOPsTable(
    device='cuda:0' if cuda_available else 'cpu',
    batch_size=1,
)

# parameters for evolutionary algorithm
target_hardware = 'flops'
latency_constraint = 400  # ms, suggested range [15, 33] ms
P = 100  # The size of population in each generation
N = 500  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    'constraint_type': target_hardware,  # Let's do FLOPs-constrained search
    'efficiency_constraint': latency_constraint,
    'mutate_prob': 0.1,  # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5,  # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': flops_lookup_table,  # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor,  # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
}

# build the evolution finder
finder = EvolutionFinder(**params)

# find optimal subnet
start = time.time()
best_valids, best_info = finder.run_evolution_search()
end = time.time()

predicted_accuracy, subnet_config, estimated_latency = best_info
print('\n*****'
      '\nBest architecture for latency constrain <= %.2f ms'
      '\nIt achieves %.2f%s predicted accuracy with %.2f ms latency.'
      '\nSubnet configuration: \n%s'
      '\nNAS took %.2f seconds.'
      '\n*****' %
      (latency_constraint, predicted_accuracy * 100, '%', estimated_latency, str(subnet_config), end - start))

# result to csv file
csv_writer.writerow(
    {'target_hardware': target_hardware,
     'latency_constraint': latency_constraint,
     'estimated_latency': estimated_latency,
     'calculation_time': end - start,
     'top1_acc': predicted_accuracy,
     'net_config': str(subnet_config)
     })
output_file.flush()

# visualize subnet config
drawing = Visualisation()
drawing.mbv3_barchart(subnet_config)
