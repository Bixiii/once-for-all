from pathlib import Path
import pickle
import csv
import time
import pickle
import random

import onnx
from onnxsim import simplify

from annette import get_database
from annette.estimation.layer_model import Layer_model
from annette.estimation.mapping_model import Mapping_model
from annette.graph import ONNXGraph, AnnetteGraph

from ofa.imagenet_classification.elastic_nn.networks import OFAResNets, OFAMobileNetV3
from ofa.nas.efficiency_predictor import AnnetteLatencyLayerPrediction, AnnetteLatencyModelResNet50

from utils import *
from result_net_configs import *

test_config = {'d': [2, 2, 2, 2, 2],
               'e': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                     0.25, 0.25],
               'w': [2, 2, 2, 2, 2, 2], 'image_size': 224}

# define parameters for OFA-network ResNet50
depth_list = [0, 1, 2]
expand_list = [0.2, 0.25, 0.35]
width_mult_list = [0.65, 0.8, 1.0]
img_size_list = []
small_input_stem = False

target_hardware = 'dnndk-mixed'  # or ncs2-mixed
lut_file_path = r'C:\Users\bixi\PycharmProjects\OnceForAllFork\LUT_ofa_resnet_dnndk-mixed.pkl'
# lut_file_path = r'C:\Users\bixi\PycharmProjects\OnceForAllFork\LUT_ofa_resnet_ncs2-mixed.pkl'

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


# build full ANNETTE LUT
def build_annette_lut(annette_model, image_sizes=None, file_path=''):
    if not image_sizes:
        image_sizes = [128, 144, 160, 176, 192, 224, 240, 256]
    latency_lut = ofa_network.build_annette_lut(image_sizes, annette_model, file_path)
    return latency_lut


# load ANNETTE LUT
def load_annette_lut(file_path):
    annette_latency_lut_file = open(file_path, 'rb')
    annette_latency_lut = pickle.load(annette_latency_lut_file)
    annette_latency_lut_file.close()
    return annette_latency_lut


# make single prediction with ANNETTE LUT
def annette_latency_lut_prediction(subnet_config, annette_latency_lut):
    start = time.time()
    latency, layer_files = ofa_network.predict_with_annette_lut(annette_latency_lut, subnet_config)
    end = time.time()
    print("Latency is " + str(latency) + ", estimation time was " + str(int((end - start) * 1000)) + "ms")
    return latency


# # make prediction form ONNX with ANNETTE for a subnet
def annette_latency_onnx_prediction(subnet_config):
    start = time.time()
    ofa_network.set_active_subnet(d=subnet_config['d'], e=subnet_config['e'], w=subnet_config['w'])
    subnet = ofa_network.get_active_subnet()
    annette_latency_predictor = AnnetteLatencyLayerPrediction()
    latency = annette_latency_predictor.predict_efficiency(subnet, (1, 3, subnet_config['image_size'], subnet_config['image_size']))
    end = time.time()
    print("Latency is " + str(latency) + ", estimation time was " + str(int((end - start) * 1000)) + "ms")
    return latency


# # evaluate ANNETTE LUT
# image_sizes = [128, 144, 160, 176, 192, 224, 240, 256]
# results_csv_file = open('evaluate_annette_lut_resnet50.csv', 'w')
# csv_writer = csv.writer(results_csv_file)
# csv_writer.writerow(['prediction from LUT', 'prediction from ANNETTE'])  # write header
#
# # for subnet_config in resnet50_flop_constrained:
# for num in range(10):
#     # get random subnet configuration and resolution
#     subnet_config = ofa_network.sample_active_subnet()
#     image_size = random.choice(image_sizes)
#     subnet_config['image_size'] = image_size
#
#     # onnx -> annette -> latency (slow)
#     annette_latency_predictor = AnnetteLatencyModelResNet50(ofa_net=ofa_network, model=target_hardware)
#     latency_from_onnx = annette_latency_predictor.predict_efficiency(subnet_config)
#
#     # annette look up table (fast)
#     latency_from_lut, _ = ofa_network.predict_with_annette_lut(annette_latency_lut, subnet_config, verify=False)
#
#     # save results to csv file
#     csv_writer.writerow([latency_from_lut, latency_from_onnx])
#
# results_csv_file.close()


# file_name = 'LUT_ofa_resnet_' + target_hardware + '.pkl'
# annette_latency_lut = load_annette_lut(file_name)
#
# subnet_config = ofa_network.sample_active_subnet()
# image_sizes = [128, 144, 160, 176, 192, 224, 240, 256]
# subnet_config['image_size'] = random.choice(image_sizes)
#
# latency, layer_files = ofa_network.predict_with_annette_lut(annette_latency_lut, subnet_config)
# print(str(layer_files))
#
# subnet = ofa_network.get_active_subnet()
# export_as_simplified_onnx(subnet, 'ofa-resnet-test-net.onnx', subnet_config['image_size'])

build_annette_lut('dnndk-mixed')