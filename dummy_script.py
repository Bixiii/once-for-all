from onnxsim import simplify
import pickle
import csv
import time

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

# # make prediction form ANNETTE-ONNX for a subnet
# image_sizes = [128, 144, 160, 176, 192, 224, 240, 256]
# image_size = 256
# ofa_network.set_active_subnet(d=test_config['d'], e=test_config['e'], w=test_config['w'])
# subnet = ofa_network.get_active_subnet()
# annette_latency_predictor = AnnetteLatencyLayerPrediction()
# latency_from_onnx = annette_latency_predictor.predict_efficiency(subnet, (1, 3, image_size, image_size))

# # build full ANNETTE LUT
# image_sizes = [128, 144, 160, 176, 192, 224, 240, 256]
# latency_lut = ofa_network.build_annette_lut(image_sizes=img_sizes)

# # load ANNETTE LUT
annette_latency_lut_file = open('full_restnet_lut.pkl', 'rb')
annette_latency_lut = pickle.load(annette_latency_lut_file)
annette_latency_lut_file.close()

# # make single prediction with ANNETTE LUT
# start = time.time()
# subnet_config = ofa_network.sample_active_subnet()
# image_size = random.choice([128, 144, 160, 176, 192, 224, 240, 256])
# subnet_config['image_size'] = image_size
# latency, subnet = ofa_network.predict_with_annette_lut(annette_latency_lut, subnet_config, verify=True)
# end = time.time()
# print("Latency is " + str(latency) + ", estimation time was " + str(int((end-start)*1000)) + " ms")

# # build single resolution ANNETTE LUT
# image_size = 256
# latency_lut = ofa_network.build_annette_lut(image_sizes=[image_size])

# # add single resolution LUT to loaded lut
# new_annette_latency_lut_file = open('new_full_restnet_lut.pkl', 'wb')
# annette_latency_lut[image_size] = latency_lut[image_size]
# pickle.dump(annette_latency_lut, new_annette_latency_lut_file)
# new_annette_latency_lut_file.close()


# # evaluate ANNETTE LUT
image_sizes = [128, 144, 160, 176, 192, 224, 240, 256]
results_csv_file = open('evaluate_annette_lut_resnet50.csv', 'w')
writer = csv.writer(results_csv_file)
writer.writerow(['prediction from LUT', 'prediction from ANNETTE'])  # write header

for _ in range(100):
    # get random subnet configuration and resolution
    subnet_config = ofa_network.sample_active_subnet()
    image_size = random.choice(image_sizes)

    # onnx -> annette -> latency (slow)
    annette_latency_predictor = AnnetteLatencyLayerPrediction()
    subnet = ofa_network.get_active_subnet()
    latency_from_onnx = annette_latency_predictor.predict_efficiency(subnet, (1, 3, image_size, image_size))

    # annette look up table (fast)
    subnet_config['image_size'] = image_size
    latency_from_lut, _ = ofa_network.predict_with_annette_lut(annette_latency_lut, subnet_config, verify=False)

    writer.writerow([latency_from_lut, latency_from_onnx])

results_csv_file.close()


