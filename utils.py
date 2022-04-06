import os
import matplotlib.pyplot as plt
import logging
from datetime import datetime

import torch
from torch.onnx import *
from ofa.utils import *


def export_layer_as_onnx(net, file_name, input_shape):
    x = torch.randn(*input_shape).cpu()
    torch.onnx.export(net, x, file_name, export_params=True)


def export_as_onnx(net, file_name, image_size=224):
    x = torch.randn(1, 3, image_size, image_size, requires_grad=True).cpu()
    torch.onnx.export(net, x, file_name, export_params=True)


def export_as_dynamic_onnx(net, file_name, image_size=224):
    x = torch.randn(1, 3, image_size, image_size, requires_grad=True).cpu()
    torch.onnx.export(net, x, file_name, export_params=True,
                      operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)


def count_flops(net, input_shape=(3, 32, 32)):
    return {'flops': count_net_flops(net, [1] + list(input_shape)) / 1e6}


def project_root():
    return os.path.dirname(os.path.abspath(__file__))


def architecture_config_2_str(architecture_config):
    config_str = ''
    if 'ks' in architecture_config:
        config_str = 'ks'
        for ks in architecture_config['ks']:
            config_str += str(ks)
    config_str += '-e'
    for e in architecture_config['e']:
        config_str += str(e)
    config_str += '-d'
    for d in architecture_config['d']:
        config_str += str(d)
    if 'r' in architecture_config:
        resolution = architecture_config['r'] if architecture_config['r'] is list else architecture_config['r'][0]
        config_str += '-r' + str(resolution)
    elif 'image_size' in architecture_config:
        config_str += '-r' + str(architecture_config['image_size'])
    return config_str


def dict_2_str(dict):
    string_representation = ''
    for [k, v] in dict.items():
        string_representation += str(k) + ': ' + str(v) + '\n'
    return string_representation


def show_pickle(fig):
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='logs/' + datetime.now().strftime('%Y%m%d_%H-%M-%S') + '.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)7s @:%(pathname)s:%(lineno)d: %(message)s'
)

# frequently used network configurations
mbv3_max_config = {
    'ks': [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    'e': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    'd': [4, 4, 4, 4, 4],
    'r': [224],
    'image_size': [224]
}

mbv3_min_config = {
    'ks': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    'e': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    'd': [2, 2, 2, 2, 2],
    'r': [160],
    'image_size': [160]
}

mbv3_random_config = {
    'ks': [3, 3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5],
    'e': [3, 4, 6, 4, 6, 6, 4, 4, 4, 3, 3, 4, 6, 4, 6, 6, 4, 4, 4, 6],
    'd': [2, 3, 4, 2, 3],
    'r': [224],
    'image_size': [224]
}


@torch.no_grad()
def count_flops_thop(network: torch.nn.Module, input_size: tuple):
    import thop
    inputs = torch.randn(*input_size, device='cuda:0')
    network = network.to('cuda:0')
    network.eval()
    # rm_bn_from_net(network)
    flops, params = thop.profile(network, (inputs,), verbose=False)
    return flops / 1e6


@torch.no_grad()
def count_flops_pthflops(network: torch.nn.Module, input_size: tuple):
    from pthflops import count_ops
    inputs = torch.randn(*input_size, device='cuda:0')
    network = network.to('cuda:0')
    network.eval()
    return count_ops(network, inputs, verbose=False)[0] / 1e6


def timestamp_string():
    time = datetime.now()
    return time.strftime('%Y%m%d_%H-%M-%S.%f')