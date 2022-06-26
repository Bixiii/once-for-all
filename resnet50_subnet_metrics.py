import csv
import pickle
import random
import onnx
from onnxsim import simplify
from pathlib import Path

# Import ANNETTE functions
from settings import annette_enabled
if annette_enabled:
    from annette import get_database
    from annette.estimation.layer_model import Layer_model
    from annette.estimation.mapping_model import Mapping_model
    from annette.graph.graph_util import ONNXGraph, AnnetteGraph

# Import common subpackages so they're available when you 'import onnx'
import onnx.checker  # noqa
import onnx.defs  # noqa
import onnx.helper  # noqa
import onnx.utils  # noqa

# Import OFA functions
from ofa.nas.efficiency_predictor import ResNet50FLOPsModel, AnnetteLatencyModel, ResNet50AnnetteLUT
from ofa.imagenet_classification.elastic_nn.networks import OFAResNets
from ofa.nas.accuracy_predictor import ResNetArchEncoder, AccuracyPredictor

from utils import *
import utils

"""
" Evaluate subnets (their configs given) with different latency estimators and with their predicted accuracy
"""

# Note: Produces correct predictions with most recent version of ANNETTE - tested with reference net
file_name_prefix = 0

# define where the output with the results should be stored
output_folder = 'resnet_subnet_metrics/'
os.makedirs(output_folder, exist_ok=True)
output_file_name = output_folder + 'metrics_ofa_resnet_subnets_' + timestamp_string() + '.csv'

# save onnx files from the estimated networks
save_onnx_files = True
save_simplified_onnx_files = True

# select evaluation metrics - comment out everything you don't want
csv_fields = [
    'net_config',
    'estimated_flops',  # flop estimator from the original repository
    'dnndk_mixed',  # latency estimations for dnndk with ANNETTE (onnx to ANNETTE)
    'ncs2_mixed',  # latency estimations for ncs2 with ANNETTE (onnx to ANNETTE)
    'dnndk_annette_lut',  # latency estimations for dnndk with look-up-table created from ANNETTE estimations
    'ncs2_annette_lut',  # latency estimations for dnndk with look-up-table created from ANNETTE estimations
    'flops_pthflops',  # flops counted pthfops package
    'flops_thop',  # flops counted with thop package
    'flops_pytorch',  # flops counted with pytorch
    'accuracy',  # estimated accuracy from original repo
    'file_id',
]

# prepare CSV-file
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

for _ in range(100):
    # get a random subnet
    subnet_config = ofa_network.sample_active_subnet()
    subnet = ofa_network.get_active_subnet()
    subnet_config['image_size'] = random.choice([128, 144, 160, 176, 192, 224, 240, 256])
    results = {'net_config': str(subnet_config)}

    # define input size
    input_size = (1, 3, subnet_config['image_size'], subnet_config['image_size'])

    # predict the flops for the subnet
    if 'estimated_flops' in csv_fields:
        flops = efficiency_predictor.get_efficiency(subnet_config)
        results['estimated_flops'] = flops

    if not annette_enabled:
        raise Exception('Annette not enabled, check settings.py')
    else:
        if ('dnndk_mixed' in csv_fields or 'ncs2_mixed' in csv_fields):
            # get ANNETTE latency estimation: export as ONNX, load ONNX for ANNETTE, make prediction
            ofa_network.set_active_subnet(d=subnet_config['d'], e=subnet_config['e'], w=subnet_config['w'])
            subnet = ofa_network.get_active_subnet()

            model_file_name = 'logs/' + timestamp_string_ms() + '.onnx'
            simplified_model_file_name = 'logs/' + timestamp_string_ms() + 'simplified.onnx'
            annette_model_file_name = 'logs/' + timestamp_string_ms() + '.json'

            export_as_onnx(subnet, model_file_name, subnet_config['image_size'])
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
        look_up_table_path = 'LUT_ofa_resnet_dnndk-mixed.pkl'
        annette_latency_lut_file = open(look_up_table_path, 'rb')
        annette_latency_lut = pickle.load(annette_latency_lut_file)
        annette_latency_lut_file.close()
        efficiency_predictor = ResNet50AnnetteLUT(ofa_network, annette_latency_lut)
        latency, _ = ofa_network.predict_with_annette_lut(annette_latency_lut, subnet_config)
        results['dnndk_annette_lut'] = latency
        print('> latency (dnndk_annette_lut): ', latency, ' ms')

    if 'ncs2_annette_lut' in csv_fields:
        look_up_table_path = 'LUT_ofa_resnet_ncs2-mixed.pkl'
        annette_latency_lut_file = open(look_up_table_path, 'rb')
        annette_latency_lut = pickle.load(annette_latency_lut_file)
        annette_latency_lut_file.close()
        efficiency_predictor = ResNet50AnnetteLUT(ofa_network, annette_latency_lut)
        latency, _ = ofa_network.predict_with_annette_lut(annette_latency_lut, subnet_config)
        results['ncs2_annette_lut'] = latency
        print('> latency (ncs2_annette_lut): ', latency, ' ms')

    # the counted FLOPs vary dependent on which library is used, approximately the are the same
    # count FLOPs with the library pthflops
    if 'flops_pthflops' in csv_fields:
        subnet = ofa_network.get_active_subnet()
        flops_pthflops = utils.count_flops_pthflops(subnet, input_size)
        results['flops_pthflops'] = flops_pthflops
        print('> FLOPs (pthflops): ', flops_pthflops, 'MFLOPs')

    # count the FLOPs with the library thop
    if 'flops_thop' in csv_fields:
        subnet = ofa_network.get_active_subnet()
        flops_thop = utils.count_flops_thop(subnet, input_size)
        results['flops_thop'] = flops_thop
        print('> FLOPs (thop): ', flops_thop, 'MFLOPs')

    # count the FLOPs with the library pytorch
    if 'flops_pytorch' in csv_fields:
        subnet = ofa_network.get_active_subnet()
        flops_pytorch = count_net_flops(subnet, input_size) / 1e6
        results['flops_pytorch'] = flops_pytorch
        print('> FLOPs (pytorch_utils): ', flops_pytorch, 'MFLOPs')

    if 'accuracy' in csv_fields:
        image_size_list = [128, 144, 160, 176, 192, 224, 240, 256]
        arch_encoder = ResNetArchEncoder(
            image_size_list=image_size_list, depth_list=ofa_network.depth_list,
            expand_list=ofa_network.expand_ratio_list,
            width_mult_list=ofa_network.width_mult_list, base_depth_list=ofa_network.BASE_DEPTH_LIST
        )
        accuracy_predictor_checkpoint_path = download_url(
            'https://hanlab.mit.edu/files/OnceForAll/tutorial/ofa_resnet50_acc_predictor.pth',
            model_dir='~/.ofa/',
        )
        accuracy_predictor = AccuracyPredictor(arch_encoder, 400, 3, checkpoint_path=accuracy_predictor_checkpoint_path,
                                               device='cpu',
                                               )
        accuracy = accuracy_predictor.predict_acc([subnet_config]).item()
        results['accuracy'] = accuracy
        print('> Accuracy: ', accuracy, '%')

    if 'file_id' in csv_fields:
        file_name_prefix += 1
        export_as_simplified_onnx(subnet, output_folder + str(file_name_prefix).zfill(6) + '.onnx', subnet_config['image_size'])
        results['file_id'] = file_name_prefix

    csv_writer.writerow(results)
    output_file.flush()

output_file.close()
