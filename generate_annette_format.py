import json
import re
import datetime
from pathlib import Path
import random

import onnx
import difflib

from onnxsim import simplify

from annette.graph import ONNXGraph

from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from ofa.utils import make_divisible
from utils import export_as_onnx


class AnnetteConverter:
    def __init__(self, template_file_path):
        # available choices
        self.img_sizes = [160, 176, 192, 208, 224]
        self.kernel_sizes = [3, 5, 7]
        self.expand_values = [3, 4, 6]
        self.depth_values = [2, 3, 4]

        # constants
        self.base_stage_width = [16, 24, 24, 24, 24, 40, 40, 40, 40, 80, 80, 80, 80, 112, 112, 112, 112, 160, 160, 160,
                                 160]
        self.se_reduction = 4  # squeeze and expand reduction
        self.divisor = 8

        json_file = json.load(open(template_file_path))
        self.network_template = json.dumps(json_file, indent=4)

    def create_annette_format(self, ks, e, d, r=224):
        start = datetime.datetime.now()
        # define fields for replacement
        replace_pattern = {
            '"img_size"': '%d' % r,
            '"img_size_2"': '%d' % (r / 2),
            '"img_size_4"': '%d' % (r / 4),
            '"img_size_8"': '%d' % (r / 8),
            '"img_size_16"': '%d' % (r / 16),
            '"img_size_32"': '%d' % (r / 32),
            '"ks_0"': '%d' % ks[0],
            '"ks_1"': '%d' % ks[1],
            '"ks_2"': '%d' % ks[2],
            '"ks_3"': '%d' % ks[3],
            '"ks_4"': '%d' % ks[4],
            '"ks_5"': '%d' % ks[5],
            '"ks_6"': '%d' % ks[6],
            '"ks_7"': '%d' % ks[7],
            '"ks_8"': '%d' % ks[8],
            '"ks_9"': '%d' % ks[9],
            '"ks_10"': '%d' % ks[10],
            '"ks_11"': '%d' % ks[11],
            '"ks_12"': '%d' % ks[12],
            '"ks_13"': '%d' % ks[13],
            '"ks_14"': '%d' % ks[14],
            '"ks_15"': '%d' % ks[15],
            '"ks_16"': '%d' % ks[16],
            '"ks_17"': '%d' % ks[17],
            '"ks_18"': '%d' % ks[18],
            '"ks_19"': '%d' % ks[19],
            '"e_0"': '%d' % (e[0] * self.base_stage_width[0]),
            '"e_1"': '%d' % (e[1] * self.base_stage_width[1]),
            '"e_2"': '%d' % (e[2] * self.base_stage_width[2]),
            '"e_3"': '%d' % (e[3] * self.base_stage_width[3]),
            '"e_4"': '%d' % (e[4] * self.base_stage_width[4]),
            '"e_5"': '%d' % (e[5] * self.base_stage_width[5]),
            '"e_6"': '%d' % (e[6] * self.base_stage_width[6]),
            '"e_7"': '%d' % (e[7] * self.base_stage_width[7]),
            '"e_8"': '%d' % (e[8] * self.base_stage_width[8]),
            '"e_9"': '%d' % (e[9] * self.base_stage_width[9]),
            '"e_10"': '%d' % (e[10] * self.base_stage_width[10]),
            '"e_11"': '%d' % (e[11] * self.base_stage_width[11]),
            '"e_12"': '%d' % (e[12] * self.base_stage_width[12]),
            '"e_13"': '%d' % (e[13] * self.base_stage_width[13]),
            '"e_14"': '%d' % (e[14] * self.base_stage_width[14]),
            '"e_15"': '%d' % (e[15] * self.base_stage_width[15]),
            '"e_16"': '%d' % (e[16] * self.base_stage_width[16]),
            '"e_17"': '%d' % (e[17] * self.base_stage_width[17]),
            '"e_18"': '%d' % (e[18] * self.base_stage_width[18]),
            '"e_19"': '%d' % (e[19] * self.base_stage_width[19]),
            '"e_4_s"': '%d' % (make_divisible(e[4] * self.base_stage_width[4] // self.se_reduction, self.divisor)),
            '"e_5_s"': '%d' % (make_divisible(e[5] * self.base_stage_width[5] // self.se_reduction, self.divisor)),
            '"e_6_s"': '%d' % (make_divisible(e[6] * self.base_stage_width[6] // self.se_reduction, self.divisor)),
            '"e_7_s"': '%d' % (make_divisible(e[7] * self.base_stage_width[7] // self.se_reduction, self.divisor)),
            '"e_12_s"': '%d' % (make_divisible(e[12] * self.base_stage_width[12] // self.se_reduction, self.divisor)),
            '"e_13_s"': '%d' % (make_divisible(e[13] * self.base_stage_width[13] // self.se_reduction, self.divisor)),
            '"e_14_s"': '%d' % (make_divisible(e[14] * self.base_stage_width[14] // self.se_reduction, self.divisor)),
            '"e_15_s"': '%d' % (make_divisible(e[15] * self.base_stage_width[15] // self.se_reduction, self.divisor)),
            '"e_16_s"': '%d' % (make_divisible(e[16] * self.base_stage_width[16] // self.se_reduction, self.divisor)),
            '"e_17_s"': '%d' % (make_divisible(e[17] * self.base_stage_width[17] // self.se_reduction, self.divisor)),
            '"e_18_s"': '%d' % (make_divisible(e[18] * self.base_stage_width[18] // self.se_reduction, self.divisor)),
            '"e_19_s"': '%d' % (make_divisible(e[19] * self.base_stage_width[19] // self.se_reduction, self.divisor)),

        }
        replace_patterns = dict((re.escape(k), v) for k, v in replace_pattern.items())
        pattern = re.compile(("|".join(replace_patterns.keys())))
        # replace fields with actual values
        annette_json = pattern.sub(lambda m: replace_patterns[re.escape(m.group(0))], self.network_template)
        end = datetime.datetime.now()
        print('Converted to ANNETTE in %.0f ms' % ((end - start).total_seconds() * 1000))
        return annette_json


#####################################
#  Test conversion (random testing)
#####################################

def compare_files(file_name_1, file_name_2):
    lines1 = (line.rstrip('\n') for line in open(file_name_1))
    lines2 = (line.rstrip('\n') for line in open(file_name_2))

    num_diff_lines = 0
    for line1, line2 in zip(lines1, lines2):
        if line1 != line2:
            num_diff_lines += 1

    # assert (num_diff_lines == 1)
    if num_diff_lines == 1:
        return True
    else:
        return False


def ofa_2_onnx_2_annette(sub_net_arch=None):
    # create annette form onnx for testing
    image_size_list = [128, 160, 192, 224]
    ofa_net = OFAMobileNetV3(ks_list=[3, 5, 7], depth_list=[4], expand_ratio_list=[3, 4, 6])
    if sub_net_arch is None:
        sub_net_arch = ofa_net.sample_active_subnet()
        sub_net_arch['r'] = random.choice(image_size_list)
    ofa_net.set_active_subnet(ks=sub_net_arch['ks'], e=sub_net_arch['e'], d=sub_net_arch['d'])
    sub_net = ofa_net.get_active_subnet()

    config_str = ''
    if sub_net_arch is not None:
        config_str = 'ks'
        for ks in sub_net_arch['ks']:
            config_str += str(ks)
        config_str += '-e'
        for e in sub_net_arch['e']:
            config_str += str(e)
        config_str += '-d'
        for d in sub_net_arch['d']:
            config_str += str(d)
        config_str += '-r' + str(sub_net_arch['r'])

    file_name = r'C:/Users/bixi/PycharmProjects/OnceForAllFork/exp/mobile_net/onnx/%s-mbv3.onnx' % config_str
    file_name_simplified = r'C:/Users/bixi/PycharmProjects/OnceForAllFork/exp/mobile_net/simplified/simplified_%s-mbv3.onnx' % config_str
    file_name_annette = r'C:/Users/bixi/PycharmProjects/OnceForAllFork/exp/mobile_net/annette/%s-mbv3.json' % config_str

    export_as_onnx(sub_net, file_name, sub_net_arch['r'])
    onnx_model = onnx.load(file_name)
    simplified, check = simplify(onnx_model)
    onnx.save(simplified, file_name_simplified)
    onnx_model_simplified = onnx.load(file_name_simplified)

    onnx_network = ONNXGraph(file_name_simplified)
    annette_graph = onnx_network.onnx_to_annette(file_name_simplified, ['input.1'], verbose=False)
    annette_graph.to_json(Path(file_name_annette))

    annette_graph_json = json.dumps(json.load(open(file_name_annette)), indent=4)
    return file_name_annette, sub_net_arch

#########################

sub_net_architectures = [
    # {'ks': [7, 3, 3, 7, 5, 5, 7, 5, 3, 3, 3, 5, 5, 3, 7, 7, 7, 3, 7, 3], 'e': [3, 6, 4, 3, 3, 3, 4, 3, 6, 3, 4, 3, 4, 6, 3, 4, 4, 4, 6, 3], 'd': [4, 4, 4, 4, 4], 'r': 128},
    # {'ks': [5, 7, 3, 3, 5, 5, 3, 3, 3, 5, 3, 3, 3, 7, 5, 7, 3, 3, 7, 5], 'e': [4, 3, 6, 6, 6, 3, 6, 4, 4, 6, 6, 6, 4, 6, 6, 6, 4, 6, 6, 3], 'd': [4, 4, 4, 4, 4], 'r': 160},
    # {'ks': [5, 5, 7, 5, 7, 5, 3, 7, 3, 3, 7, 7, 3, 3, 3, 5, 5, 7, 3, 7], 'e': [6, 3, 3, 6, 3, 3, 6, 6, 6, 3, 6, 4, 4, 4, 4, 4, 6, 3, 4, 6], 'd': [4, 4, 4, 4, 4], 'r': 160},
    {'ks': [3, 5, 7, 3, 3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 5, 3, 7, 5, 7, 5], 'e': [3, 4, 6, 4, 6, 6, 4, 4, 4, 3, 3, 4, 6, 4, 6, 6, 4, 4, 4, 6], 'd': [4, 4, 4, 4, 4], 'r': 192},
]

logfile_name = 'test_results_converter.txt'
logfile = open(logfile_name, 'w')
error_file_name = 'errors_converter.txt'
error_file = open(error_file_name, 'w')

# for sub_net_arch in sub_net_architectures:
for i in range(10):
    logfile.write('***Test***\n')
    file_name_annette, sub_net_arch = ofa_2_onnx_2_annette()
    # file_name_annette, _ = ofa_2_onnx_2_annette(sub_net_arch)
    logfile.write(str(sub_net_arch))

    # set up converter
    json_network_template_path = r'C:\Users\bixi\PycharmProjects\OnceForAllFork\exp\mobile_net\annette\generalized_mbv3.json'
    converter = AnnetteConverter(json_network_template_path)

    # do conversion
    annette_json = converter.create_annette_format(**sub_net_arch)

    # save file
    file_name_converted_annette = 'mbv3.json'
    new_json_file = open(file_name_converted_annette, 'w')
    new_json_file.write(annette_json)
    new_json_file.close()

    match = compare_files(file_name_annette, file_name_converted_annette)

    if match:
        logfile.write('\nConversion is correct :)\n')
        print('\nConversion is correct :)\n')
    else:
        logfile.write('\n>>>>>> !!! ERROR !!! >>>>>>\n')
        error_file.write(str(sub_net_arch))
        print('\n>>>>>> !!! ERROR !!! >>>>>>\n')

print('~~~* fin *~~~')
