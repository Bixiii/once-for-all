import time
import argparse

from ofa.imagenet_classification.elastic_nn.networks import OFAResNets

from ofa.nas.efficiency_predictor import ResNet50LatencyTable, ResNet50FLOPsModel

from ofa.nas.accuracy_predictor import ResNetArchEncoder, AccuracyPredictor

from ofa.nas.search_algorithm import EvolutionFinder

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='ResNet50', choices=['ResNet50'])
parser.add_argument('--acc_pred_path', default='', help='Path to the trained accuracy predictor')
args = parser.parse_args()

# NAS constrains
target_hardware = 'flops'
latency_constraint = 100

# Accuracy predictor
args.n_layers = 1
args.hidden_size = 100

# OFA network settings
args.image_size_list = [32]
args.depth_list = [0, 1, 2]
args.expand_ratio_list = [0.2, 0.25, 0.35]
args.width_mult_list = [0.65, 0.8, 1.0]
args.small_input_stem = True

# Dataset
args.num_classes = 10

arch_encoder = ResNetArchEncoder(
    image_size_list=args.image_size_list,
    depth_list=args.depth_list,
    expand_list=args.expand_ratio_list,
    width_mult_list=args.width_mult_list,
    small_input_stem=args.small_input_stem,
)
accuracy_predictor = AccuracyPredictor(
    arch_encoder=arch_encoder,
    n_layers=args.n_layers,
    hidden_size=args.hidden_size,
    checkpoint_path=args.accuracy_predictor_path,
)

ofa_network = OFAResNets(
    n_classes=args.num_classes,
    dropout_rate=0,
    depth_list=args.depth_list,
    expand_ratio_list=args.expand_ratio_list,
    width_mult_list=args.width_mult_list,
    small_input_stem=args.small_input_stem,
    dataset=args.dataset
)

efficiency_predictor = ResNet50FLOPsModel(ofa_network)

# Hyper-parameters for the evolutionary search process
P = 50  # The size of population in each generation
N = 250  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
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
result_lis = []
st = time.time()
best_valid, best_info = finder.run_evolution_search(latency_constraint)
result_lis.append(best_info)
ed = time.time()
print('Found best architecture on %s with latency <= %.2f ms in %.2f seconds! '
      'It achieves %.2f%s predicted accuracy with %.2f ms latency on %s.' 
      '\nArchitecture configuration: %s' %
      (target_hardware, latency_constraint, ed-st, best_info[0] * 100, '%', best_info[-1], target_hardware,
       str(best_info[1])))
