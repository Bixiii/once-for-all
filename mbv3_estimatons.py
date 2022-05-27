import argparse
import torch
import random
import numpy as np
import csv

from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from ofa.nas.efficiency_predictor import AnnetteLatencyModel
from ofa.tutorial import AccuracyPredictor, LatencyTable, FLOPsTable
import utils
from ofa.utils.pytorch_utils import count_net_flops

"""
Use latency and accuracy predictor on MobileNetV3 architecture
"""

network_configs = utils.mbv3_min_config

# CSV
csv_fields = [
    'net_config',
    # 'measured_acc',
    'predicted_acc',
    'estimated_flops',
    'estimated_latency',
    'annette_dnndk_mixed',
    'flops_pthflops',
    'flops_thop',
    'flops_pytorch',
]

output_file_name = './logs/ofa_optimized_subnets_add_info.csv'
output_file = open(output_file_name, 'w', newline='')
csv_writer = csv.DictWriter(output_file, fieldnames=csv_fields)
csv_writer.writeheader()

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

#######################################
if not isinstance(network_configs, list):
    network_configs = [network_configs]

for network_config in network_configs:

    results = {'net_config': str(network_config)}

    ofa_network.set_active_subnet(ks=network_config['ks'], e=network_config['e'],
                                  d=network_config['d'])
    subnet = ofa_network.get_active_subnet()
    input_size = (1, 3, network_config['r'][0], network_config['r'][0])

    if 'predicted_acc' in csv_fields:
        # accuracy predictor (pretrained from OFA repo)
        accuracy_predictor = AccuracyPredictor(
            pretrained=True,
            device='cuda:0' if cuda_available else 'cpu'
        )
        predicted_acc = accuracy_predictor.predict_accuracy(network_config).item()
        results['predicted_acc'] = predicted_acc
        print('> Estimated accuracy: ', predicted_acc)

    # efficiency predictor
    if 'estimated_flops' in csv_fields:
        efficiency_predictor = FLOPsTable(
            device='cuda:0' if cuda_available else 'cpu',
            batch_size=1,
            pred_type='flops',
        )
        estimated_flops = efficiency_predictor.predict_efficiency(network_config)
        results['estimated_flops'] = estimated_flops
        print('> Efficiency: (estimated flops)', estimated_flops)

    if 'estimated_latency' in csv_fields:
        efficiency_predictor = FLOPsTable(
            device='cuda:0' if cuda_available else 'cpu',
            batch_size=1,
            pred_type='latency',
        )
        estimated_latency = efficiency_predictor.predict_efficiency(network_config)
        results['estimated_latency'] = estimated_latency
        print('> Efficiency (local estimated latency): ', estimated_latency)

    if 'annette_dnndk_mixed' in csv_fields:
        efficiency_predictor = AnnetteLatencyModel(ofa_network, model='dnndk-mixed')
        annette_dnndk_mixed = efficiency_predictor.predict_efficiency(network_config)
        results['annette_dnndk_mixed'] = annette_dnndk_mixed
        print('> Efficiency (annette-dnndk-mixed): ', annette_dnndk_mixed)

    # count flops
    if 'flops_pthflops' in csv_fields:
        flops_pthflops = utils.count_flops_pthflops(subnet, input_size)
        results['flops_pthflops'] = flops_pthflops
        print('> FLOPs (pthflops): ', flops_pthflops, 'MFLOPs')

    if 'flops_thop' in csv_fields:
        flops_thop = utils.count_flops_thop(subnet, input_size)
        results['flops_thop'] = flops_thop
        print('> FLOPs (thop): ', flops_thop, 'MFLOPs')

    if 'flops_pytorch' in csv_fields:
        flops_pytorch = count_net_flops(subnet, input_size) / 1e6
        results['flops_pytorch'] = flops_pytorch
        print('> FLOPs (pytorch_utils): ', flops_pytorch, 'MFLOPs')

    # measure actual accuracy
    if 'measured_acc' in csv_fields:
        path_ofa_net = 'pretrained/ofa_nets/ofa_mbv3_d234_e346_k357_w1.0'
        init = torch.load(path_ofa_net, map_location='cpu')['state_dict']
        ofa_network.load_state_dict(init)

        data_path = 'datasets/imagenet_1k'
        run_config = ImagenetRunConfig(test_batch_size=1, n_worker=0, dataset='imagenet', data_path=data_path)
        run_manager = RunManager('.tmp/eval_subnet', ofa_network, run_config, init=False)

        val_dataset = []
        run_manager.run_config.data_provider.assign_active_img_size(network_config['r'][0])
        for images, labels in run_manager.run_config.valid_loader:
            val_dataset.append((images, labels))
        run_manager.reset_running_statistics(ofa_network)
        loss, (top1, top5) = run_manager.validate(net=ofa_network, data_loader=val_dataset, no_logs=True)
        results['measured_acc'] = top1
        print('> Accuracy on 1k Imagenet Subset: ', top1)

    csv_writer.writerow(results)
    output_file.flush()

output_file.close()