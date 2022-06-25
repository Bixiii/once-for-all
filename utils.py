import os
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import re

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


def export_as_simplified_onnx(net, file_name, image_size=224):
    from onnx import load, save
    from onnxsim import simplify
    simplified_model_file_name = file_name[:-5] + '_simplified.onnx'
    export_as_onnx(net, file_name, image_size)
    onnx_model = load(file_name)
    simplified_model, check = simplify(onnx_model)
    save(simplified_model, simplified_model_file_name)
    print('Saved as: ' + simplified_model_file_name)
    os.remove(file_name)


def export_pytorch_state_dict(net, file_name, image_size=224):
    raise NotImplementedError


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


# Returns a string representation of a ofa-subnet-configuration without spaces or dots
def subnet_config_2_file_name(subnet_config):
    string_representation = ''
    for [k, v] in subnet_config.items():
        string_content_value = ''
        if isinstance(v, list):
            for value in v:
                if isinstance(value, float):
                    value = int(value * 100)
                string_content_value = string_content_value + str(value)
        else:
            string_content_value = string_content_value + str(v)

        string_representation += str(k) + string_content_value + '_'
    return string_representation[:-1]


def show_pickle(fig):
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


logger = logging.getLogger(__name__)
log_directory = os.path.dirname(__file__) + '/logs/'
os.makedirs(log_directory, exist_ok=True)
log_file_name = log_directory + datetime.now().strftime('%Y%m%d_%H-%M-%S') + '.log'
file_handler = logging.FileHandler(log_file_name, mode='w', encoding=None, delay=False)
logging.basicConfig(
    filename=log_file_name,
    level=logging.INFO,
    format='%(asctime)s %(levelname)7s @:%(pathname)s:%(lineno)d: %(message)s'
)


# frequently used test network configurations
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

resnet_max_config = {
    'd': [2, 2, 2, 2, 2],
    'e': [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35],
    'w': [2, 2, 2, 2, 2, 2],
    'image_size': 224
}

resnet_min_config = {
    'd': [0, 0, 0, 0, 0],
    'e': [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
    'w': [0, 0, 0, 0, 0, 0],
    'image_size': 224
}


@torch.no_grad()
def count_flops_thop(network: torch.nn.Module, input_size: tuple):
    import thop
    inputs = torch.randn(*input_size, device='cpu')
    network.eval()
    # rm_bn_from_net(network)
    flops, params = thop.profile(network, (inputs,), verbose=False)
    return flops / 1e6


@torch.no_grad()
def count_flops_pthflops(network: torch.nn.Module, input_size: tuple):
    from pthflops import count_ops
    inputs = torch.randn(*input_size, device='cpu')
    network.eval()
    return count_ops(network, inputs, verbose=False)[0] / 1e6


def timestamp_string():
    time = datetime.now()
    return time.strftime('%Y%m%d_%H-%M-%S')


def timestamp_string_ms():
    time = datetime.now()
    return time.strftime('%Y%m%d_%H-%M-%S-%f')


def file_id():
    file_id.id += 1
    return str(file_id.id).zfill(6)


file_id.id = 0


def pickle2csv(file_name):
    import pickle
    import pandas
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    data_frame = pandas.DataFrame(data)
    data_frame.to_csv(file_name[:file_name.find('.')] + '.csv')


def string_2_list(list_as_string):
    """
    converts a string representation from a python list to an actual list
    Args:
        list_as_string (): string representation with formatting like '[element_1, ..., element_n]'

    Returns: list of strings

    """
    assert(list_as_string[0] == '[')
    assert(list_as_string[-1] == ']')
    new_list = list_as_string[1:-1].split(", ")
    return new_list


def string_2_dict(dict_as_string):
    assert(dict_as_string[0] == '{')
    assert(dict_as_string[-1] == '}')
    new_dict = {}
    key_pattern = re.compile("\'.+?\'")
    keys = key_pattern.findall(dict_as_string)
    value_patter = re.compile("(\: .+?, \')")
    values = value_patter.findall(dict_as_string)
    for i, key in enumerate(keys[:-1]):
        new_dict[key[1:-1]] = values[i][2:-3]
    last_value = dict_as_string[dict_as_string.rfind("\'")+3:-1]
    new_dict[keys[-1][1:-1]] = last_value
    return new_dict


def string_2_config_dict(config_dict_as_string):
    config = string_2_dict(config_dict_as_string)
    for key, value in config.items():
        if value[0] == '[':
            value = string_2_list(value)
            for i, element in enumerate(value):
                if '.' in element:
                    value[i] = float(element)
                else:
                    value[i] = int(element)
        else:
            value = int(value)

        config[key] = value
    return config
