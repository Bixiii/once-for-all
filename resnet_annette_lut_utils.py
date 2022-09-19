import csv
import time
import pickle
import random

from ofa.imagenet_classification.elastic_nn.networks import OFAResNets
from ofa.nas.efficiency_predictor import AnnetteLatencyLayerPrediction, AnnetteLatencyModelResNet50

from utils import *

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
image_sizes = [128, 144, 160, 176, 192, 224, 240, 256]

target_hardware = 'dnndk-mixed'  # or ncs2-mixed
path_annette_lut = 'LUT_ofa_resnet_' + target_hardware + '.pkl'

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
def build_annette_lut(annette_model, file_path=''):
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
    latency = annette_latency_predictor.predict_efficiency(subnet, (
    1, 3, subnet_config['image_size'], subnet_config['image_size']))
    end = time.time()
    print("Latency is " + str(latency) + ", estimation time was " + str(int((end - start) * 1000)) + "ms")
    return latency


def test_annette_lut():
    """
    Make a latency prediction with the annette-lut and print the file ids used from the annette-lut
    """
    annette_latency_lut = load_annette_lut(path_annette_lut)

    subnet_config = ofa_network.sample_active_subnet()
    subnet_config['image_size'] = random.choice(image_sizes)

    latency, layer_files = ofa_network.predict_with_annette_lut(annette_latency_lut, subnet_config)
    print(str(layer_files))

    subnet = ofa_network.get_active_subnet()
    export_as_simplified_onnx(subnet, 'ofa-resnet-test-net.onnx', subnet_config['image_size'])


def evaluate_annette_lut():
    """
    Make predictions with annette-lut and predictions with annette from onnx files
    Write results to csv file
    """
    results_csv_file = open('evaluate_annette_lut_resnet50.csv', 'w')
    csv_writer = csv.writer(results_csv_file)
    csv_writer.writerow(['prediction from LUT', 'prediction from ANNETTE'])  # write header

    annette_latency_lut = load_annette_lut(path_annette_lut)

    # for subnet_config in resnet50_flop_constrained:
    for num in range(1):
        # get random subnet configuration and resolution
        subnet_config = ofa_network.sample_active_subnet()
        image_size = random.choice(image_sizes)
        subnet_config['image_size'] = image_size

        # onnx -> annette -> latency (slow)
        annette_latency_predictor = AnnetteLatencyModelResNet50(ofa_net=ofa_network, model=target_hardware)
        latency_from_onnx = annette_latency_predictor.predict_efficiency(subnet_config)

        # annette look up table (fast)
        latency_from_lut, _ = ofa_network.predict_with_annette_lut(annette_latency_lut, subnet_config)

        # save results to csv file
        csv_writer.writerow([latency_from_lut, latency_from_onnx])

    results_csv_file.close()
