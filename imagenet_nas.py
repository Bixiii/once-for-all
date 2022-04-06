import argparse
import torch
import time
import torch
import random
import numpy as np
import csv
import os
import pickle

from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3, OFAResNets

from ofa.nas.efficiency_predictor import AnnetteLatencyModel, AnnetteLatencyModelResNet50

from visualize_subnet import Visualisation
from utils import architecture_config_2_str, logger, dict_2_str
from ofa.utils import download_url

"""
NAS for Imagenet using pretrained networks from OnceForAll publication
"""


def mobilenet_predictors(population_size, max_time_budget, parent_ratio, constraint_type):

    from ofa.tutorial import EvolutionFinder
    from ofa.tutorial import AccuracyPredictor, LatencyTable, FLOPsTable

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

    # accuracy predictor (pretrained from OFA repo)
    accuracy_predictor = AccuracyPredictor(
        pretrained=True,
        device='cuda:0' if cuda_available else 'cpu'
    )

    # efficiency predictor
    if constraint_type == 'flops' or constraint_type == 'latency' or constraint_type == 'annette':
        efficiency_predictor = FLOPsTable(
            device='cuda:0' if cuda_available else 'cpu',
            batch_size=1,
            pred_type=constraint_type,
            annette_model=annette_model
        )
    else:
        raise NotImplementedError

    evolution_parameters = {
        'constraint_type': constraint_type,
        'mutate_prob': 0.1,  # The probability of mutation in evolutionary search
        'mutation_ratio': 0.5,  # The ratio of networks that are generated through mutation in generation n >= 2.
        'efficiency_predictor': efficiency_predictor,  # To use a predefined efficiency predictor.
        'accuracy_predictor': accuracy_predictor,  # To use a predefined accuracy_predictor predictor.
        'population_size': population_size,
        'max_time_budget': max_time_budget,
        'parent_ratio': parent_ratio,
    }

    evolution_finder = EvolutionFinder(**evolution_parameters)

    return ofa_network, evolution_finder, efficiency_predictor


def resnet_predictors(population_size, max_time_budget, parent_ratio, constraint_type):

    from ofa.nas.search_algorithm import EvolutionFinder
    from ofa.nas.accuracy_predictor import AccuracyPredictor, ResNetArchEncoder
    from ofa.nas.efficiency_predictor import ResNet50FLOPsModel

    # OFA network
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

    # accuracy predictor (pretrained from OFA repo)

    image_size_list = [128, 144, 160, 176, 192, 224, 240, 256]
    arch_encoder = ResNetArchEncoder(
        image_size_list=image_size_list, depth_list=ofa_network.depth_list, expand_list=ofa_network.expand_ratio_list,
        width_mult_list=ofa_network.width_mult_list, base_depth_list=ofa_network.BASE_DEPTH_LIST
    )

    accuracy_predictor_checkpoint_path = download_url(
        'https://hanlab.mit.edu/files/OnceForAll/tutorial/ofa_resnet50_acc_predictor.pth',
        model_dir='~/.ofa/',
    )
    accuracy_predictor = AccuracyPredictor(arch_encoder, 400, 3, checkpoint_path=accuracy_predictor_checkpoint_path,
                                           device='cpu',
                                           )

    # efficiency predictor
    if constrain_type == 'flops':
        efficiency_predictor = ResNet50FLOPsModel(ofa_network)
    elif constraint_type == 'annette':
        efficiency_predictor = AnnetteLatencyModelResNet50(ofa_network, annette_model)
    else:
        raise NotImplementedError

    evolution_parameters = {
        'constraint_type': constraint_type,
        'mutate_prob': 0.1,  # The probability of mutation in evolutionary search
        'mutation_ratio': 0.5,  # The ratio of networks that are generated through mutation in generation n >= 2.
        'efficiency_predictor': efficiency_predictor,  # To use a predefined efficiency predictor.
        'accuracy_predictor': accuracy_predictor,  # To use a predefined accuracy_predictor predictor.
        'population_size': population_size,
        'max_time_budget': max_time_budget,
        'parent_ratio': parent_ratio,
    }

    # TODO
    # self.arch_mutate_prob = kwargs.get('arch_mutate_prob', 0.1)
    # self.resolution_mutate_prob = kwargs.get('resolution_mutate_prob', 0.5)

    evolution_finder = EvolutionFinder(**evolution_parameters)

    return ofa_network, evolution_finder


parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, choices=['ResNet50', 'MobileNetV3'], help='Network architecture')
parser.add_argument('--latency', type=float, help='Latency constrain')
parser.add_argument('--constrain_type', type=str, choices=['annette', 'flops', 'latency'],
                    help='Mechanism used for latency estimation')
parser.add_argument('--annette_model', type=str, default=None, choices=AnnetteLatencyModel.layers,
                    help='Select which model should be used for ANNETTE latency estimation')
args = parser.parse_args()

# net = args.net
# latency_constraints = args.latency
# constrain_type = args.constrain_type
# annette_model = args.annette_model
# TODO remove local definitions before committing
net = 'MobileNetV3'
latency_constraints = [40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20]
constrain_type = 'annette'
annette_model = 'ncs2-mixed'

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
results_file_name = parent_folder + constrain_type + '_optimized_' + net + '_subnets.csv'
print('File with results from NAS created <' + results_file_name + '>.')
write_header = False if os.path.exists(results_file_name) else True
output_file = open(results_file_name, 'a+', newline='')
csv_data_fields = ['constrain_type', 'target_hardware', 'latency_constraint', 'estimated_latency', 'calculation_time',
                   'top1_acc', 'net_config']
csv_writer = csv.DictWriter(output_file, fieldnames=csv_data_fields)
if write_header:
    csv_writer.writeheader()

# parameters for evolutionary algorithm
P = 75  # The size of population in each generation
N = 250  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation

if net == 'MobileNetV3':
    ofa_network, evolution_finder, efficency_predictor = mobilenet_predictors(P, N, r, constrain_type)
elif net == 'ResNet50':
    ofa_network, evolution_finder = resnet_predictors(P, N, r, constrain_type)  # TODO here I need to integrate Annette
else:
    raise NotImplementedError

if not isinstance(latency_constraints, list):
    latency_constraints = [latency_constraints]

for latency_constraint in latency_constraints:

    # find optimal subnet
    start = time.time()
    logger.info('Start evolution search')
    best_valids, best_subnet_info = evolution_finder.run_evolution_search(efficiency_constraint=latency_constraint)
    logger.info('Finished evolution search')
    end = time.time()

    predicted_accuracy, subnet_config, estimated_latency = best_subnet_info
    info_string = ('\n*****'
                   '\nBest architecture for latency constrain <= %.2f ms'
                   '\nIt achieves %.2f%s predicted accuracy with %.2f ms latency.'
                   '\nSubnet configuration: \n%s'
                   '\nNAS took %.2f seconds.'
                   '\n*****' %
                   (latency_constraint, predicted_accuracy * 100, '%', estimated_latency, str(subnet_config),
                    end - start))
    logger.info(info_string)
    print(info_string)

    # result to csv file
    csv_writer.writerow(
        {'constrain_type': constrain_type,
         'target_hardware': annette_model,
         'latency_constraint': latency_constraint,
         'estimated_latency': estimated_latency,
         'calculation_time': end - start,
         'top1_acc': predicted_accuracy,
         'net_config': str(subnet_config)
         })
    output_file.flush()

    # visualize subnet config
    # drawing = Visualisation()
    # title = (('Constrain: %.2f %s \nAccuracy: %.2f' % (
    #     estimated_latency, 'MFLOPs' if constrain_type == 'flops' else 'ms', predicted_accuracy * 100)) + '%')
    #
    # fig = drawing.mbv3_barchart(subnet_config,
    #                             save_path=parent_folder + (architecture_config_2_str(subnet_config) + '.png'),
    #                             title=title, show_fixed=True, relative=False, show=False)
    # pickle.dump(fig, open(parent_folder + architecture_config_2_str(subnet_config) + '.pickle', 'wb'))
    #
    # fig_rel = drawing.mbv3_barchart(subnet_config,
    #                                 save_path=parent_folder + (
    #                                             architecture_config_2_str(subnet_config) + '_relative.png'),
    #                                 title=title, show_fixed=True, relative=True, show=True)
    # pickle.dump(fig_rel, open(parent_folder + architecture_config_2_str(subnet_config) + '_relative.pickle', 'wb'))
