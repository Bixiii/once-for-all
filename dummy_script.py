from onnxsim import simplify
import pickle
import csv

from ofa.imagenet_classification.elastic_nn.networks import OFAResNets, OFAMobileNetV3
import onnx

from ofa.nas.efficiency_predictor import AnnetteLatencyLayerPrediction
from utils import timestamp_string, export_as_onnx, architecture_config_2_str, export_layer_as_onnx
from result_net_configs import *
import pickle
import random

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

image_sizes = [128, 160, 192, 224]  # TODO build single LUT for all resolution - ResNet has more
image_size = 224

# ofa_network.set_active_subnet(d=test_config['d'], e=test_config['e'], w=test_config['w'])
# subnet = ofa_network.get_active_subnet()
# annette_latency_predictor = AnnetteLatencyLayerPrediction()
# latency_from_onnx = annette_latency_predictor.predict_efficiency(subnet, (1, 3, image_size, image_size))


# latency_lut = ofa_network.build_annette_lut(image_sizes=image_sizes)

# lut_file = open('tmp_lut.pkl', 'wb')
# pickle.dump(latency_lut, lut_file)
# lut_file.close()

annette_latency_lut_file = open('full_restnet_lut.pkl', 'rb')
annette_latency_lut = pickle.load(annette_latency_lut_file)

results_csv_file = open('evaluate_annette_lut_resnet50.csv', 'w')
writer = csv.writer(results_csv_file)
writer.writerow(['prediction from LUT', 'prediction from ANNETTE'])  # write header

for _ in range(100):
    # get random subnet configuration and resolution
    subnet_config = ofa_network.sample_active_subnet()
    image_size = random.choice(image_sizes)
    subnet_config['image_size'] = image_size

    # make latency predictions
    latency_from_lut, subnet = ofa_network.predict_with_annette_lut(annette_latency_lut, subnet_config, verify=True)
    annette_latency_predictor = AnnetteLatencyLayerPrediction()
    latency_from_onnx = annette_latency_predictor.predict_efficiency(subnet, (1, 3, image_size, image_size))  # TODO we want resolution
    writer.writerow([latency_from_lut, latency_from_onnx])

results_csv_file.close()


