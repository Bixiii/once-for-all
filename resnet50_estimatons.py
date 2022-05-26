import csv
import pickle

from annette import get_database
from annette.estimation.layer_model import Layer_model
from annette.estimation.mapping_model import Mapping_model

from ofa.nas.efficiency_predictor import ResNet50FLOPsModel, AnnetteLatencyModel, ResNet50AnnetteLUT
from ofa.imagenet_classification.elastic_nn.networks import OFAResNets
from utils import *

# Import common subpackages so they're available when you 'import onnx'
import onnx.checker  # noqa
import onnx.defs  # noqa
import onnx.helper  # noqa
import onnx.utils  # noqa

import onnx
from onnxsim import simplify
from pathlib import Path
from result_net_configs import resnet50_flop_constrained, resnet50_dnndk_constrained, resnet50_ncs2_constrained

import utils

from annette.graph.graph_util import ONNXGraph, AnnetteGraph
import random

from ofa.tutorial import FLOPsTable

"""
" Evaluate subnets (their config has to be given) with different latency estimators and with their predicted accuracy
"""

# Note: Produces correct predictions with most recent version of ANNETTE - tested with reference net

# select evaluation metrics - comment out everything you don't want

csv_fields = [
    'net_config',
    # 'estimated_flops',
    'dnndk_mixed',
    'ncs2_mixed',
    'dnndk_annette_lut',
    'ncs2_annette_lut',
    # 'flops_pthflops',
    # 'flops_thop',
    # 'flops_pytorch',
]

# # define where the output CSV-file with the results should be stored
output_file_name = './logs/ofa_resnet_random_subnets_add_info.csv'
# # prepare CSV-file
output_file = open(output_file_name, 'w', newline='')
csv_writer = csv.DictWriter(output_file, fieldnames=csv_fields)
csv_writer.writeheader()

# define parameters for OFA-network ResNet50
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

# crate efficiency predictor to estimate FLOPs
efficiency_predictor = None
if 'estimated_flops' in csv_fields:
    efficiency_predictor = ResNet50FLOPsModel(ofa_network)

for _ in range(10):
    # get a random subnet
    subnet_config = ofa_network.sample_active_subnet()
    subnet_config['image_size'] = random.choice([128, 144, 160, 176, 192, 224, 240, 256])
    results = {'net_config': str(subnet_config)}

    # define input size
    input_size = (1, 3, subnet_config['image_size'], subnet_config['image_size'])

    # predict the flops for the subnet
    if 'estimated_flops' in csv_fields:
        flops = efficiency_predictor.get_efficiency(subnet_config)
        results['estimated_flops'] = flops

    if 'dnndk_mixed' in csv_fields or 'ncs2_mixed' in csv_fields:
        # get ANNETTE latency estimation: export as ONNX, load ONNX for ANNETTE, make prediction
        ofa_network.set_active_subnet(d=subnet_config['d'], e=subnet_config['e'], w=subnet_config['w'])
        subnet = ofa_network.get_active_subnet()

        model_file_name = 'logs/' + timestamp_string() + '.onnx'
        simplified_model_file_name = 'logs/' + timestamp_string() + 'simplified.onnx'
        annette_model_file_name = 'logs/' + timestamp_string() + '.json'
        export_as_onnx(subnet, model_file_name)

        onnx_model = onnx.load(model_file_name)
        simplified_model, check = simplify(onnx_model)
        onnx.save(simplified_model, simplified_model_file_name)

        onnx_network = ONNXGraph(simplified_model_file_name)
        annette_graph = onnx_network.onnx_to_annette(simplified_model_file_name, ['input.1'], name_policy='renumerate')
        json_file = Path(annette_model_file_name)
        annette_graph.to_json(json_file)

        model = AnnetteGraph('ofa-net', annette_model_file_name)

        if 'dnndk_mixed' in csv_fields:
            # load ANNETTE models
            mapping = 'dnndk'
            layer = 'dnndk-mixed'
            opt = Mapping_model.from_json(get_database('models', 'mapping', mapping + '.json'))
            mod = Layer_model.from_json(get_database('models', 'layer', layer + '.json'))

            opt.run_optimization(model)
            res = mod.estimate_model(model)

            results['dnndk_mixed'] = res[0]
            print('> latency (dnndk-mixed): ', res[0], ' ms')

        if 'ncs2_mixed' in csv_fields:
            # load ANNETTE models
            mapping = 'ov2'
            layer = 'ncs2-mixed'
            opt = Mapping_model.from_json(get_database('models', 'mapping', mapping + '.json'))
            mod = Layer_model.from_json(get_database('models', 'layer', layer + '.json'))

            opt.run_optimization(model)
            res = mod.estimate_model(model)

            results['ncs2_mixed'] = res[0]
            print('> latency (ncs2-mixed): ', res[0], ' ms')

    # use look-up-table for annette latency prediction
    if 'dnndk_annette_lut' in csv_fields:
        look_up_table_path = 'restnet_dnndk_lut.pkl'
        annette_latency_lut_file = open(look_up_table_path, 'rb')
        annette_latency_lut = pickle.load(annette_latency_lut_file)
        annette_latency_lut_file.close()
        efficiency_predictor = ResNet50AnnetteLUT(ofa_network, annette_latency_lut)
        latency, subnet = ofa_network.predict_with_annette_lut(annette_latency_lut, subnet_config, verify=False)
        results['dnndk_annette_lut'] = latency
        print('> latency (dnndk_annette_lut): ', latency, ' ms')

    if 'ncs2_annette_lut' in csv_fields:
        look_up_table_path = r'C:\Users\bixi\PycharmProjects\OnceForAllFork\restnet_ncs2_lut.pkl'
        annette_latency_lut_file = open(look_up_table_path, 'rb')
        annette_latency_lut = pickle.load(annette_latency_lut_file)
        annette_latency_lut_file.close()
        efficiency_predictor = ResNet50AnnetteLUT(ofa_network, annette_latency_lut)
        latency, subnet = ofa_network.predict_with_annette_lut(annette_latency_lut, subnet_config, verify=False)
        results['ncs2_annette_lut'] = latency
        print('> latency (ncs2_annette_lut): ', latency, ' ms')

    # the counted FLOPs vary dependent on which library is used, approximately the are the same
    # count FLOPs with the library pthflops
    if 'flops_pthflops' in csv_fields:
        flops_pthflops = utils.count_flops_pthflops(subnet, input_size)
        results['flops_pthflops'] = flops_pthflops
        print('> FLOPs (pthflops): ', flops_pthflops, 'MFLOPs')

    # count the FLOPs with the library thop
    if 'flops_thop' in csv_fields:
        flops_thop = utils.count_flops_thop(subnet, input_size)
        results['flops_thop'] = flops_thop
        print('> FLOPs (thop): ', flops_thop, 'MFLOPs')

    # count the FLOPs with the library pytorch
    if 'flops_pytorch' in csv_fields:
        flops_pytorch = count_net_flops(subnet, input_size) / 1e6
        results['flops_pytorch'] = flops_pytorch
        print('> FLOPs (pytorch_utils): ', flops_pytorch, 'MFLOPs')

    csv_writer.writerow(results)
    output_file.flush()

output_file.close()
