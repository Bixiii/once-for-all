import csv

from annette import get_database
from annette.estimation.layer_model import Layer_model
from annette.estimation.mapping_model import Mapping_model

from ofa.nas.efficiency_predictor import ResNet50FLOPsModel
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

from annette.graph.graph_util import ONNXGraph, AnnetteGraph
import random

"""
" Evaluate subnets (their config has to be given) with different latency estimators and with their predicted accuracy
"""

csv_fields = [
    'net_config',
    # 'measured_acc',
    # 'predicted_acc',
    'estimated_flops',
    # 'estimated_latency',
    'annette_dnndk_mixed',
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

################################
# configure evaluation metrics #
################################

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
efficiency_predictor = ResNet50FLOPsModel(ofa_network)

for _ in range(100):
    # get a random subnet
    random_config = ofa_network.sample_active_subnet()
    random_config['image_size'] = image_size = random.choice([128, 160, 192, 224])

    results = {'net_config': str(random_config)}

    # predict the flops for the subnet
    flops = efficiency_predictor.get_efficiency(random_config)
    results['estimated_flops'] = flops

    # get ANNETTE latency estimation: export as ONNX, load ONNX for ANNETTE, make prediction
    subnet = ofa_network.get_active_subnet(
        ofa_network.set_active_subnet(d=random_config['d'], e=random_config['e'], w=random_config['w']))

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

    # load ANNETTE models
    mapping = 'dnndk'
    layer = 'dnndk-mixed'

    opt = Mapping_model.from_json(get_database('models', 'mapping', mapping + '.json'))
    mod = Layer_model.from_json(get_database('models', 'layer', layer + '.json'))

    model = AnnetteGraph('ofa-net', annette_model_file_name)

    opt.run_optimization(model)
    res = mod.estimate_model(model)

    results['annette_dnndk_mixed'] = res[0]

    # logger.debug('Finished ANNETTE efficiency prediction')

    print('arch config: ' + str(random_config) + ' FLOPs: ' + str(flops) + ' ANNETTE dnndk-mixed latency: ' +
          str(res[0]))

    csv_writer.writerow(results)
    output_file.flush()

output_file.close()
