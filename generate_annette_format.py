import json
import os
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

        self.json_file = json.load(open(template_file_path))
        # self.network_template = json.dumps(self.json_file, indent=4)

        # first group
        self.stage2_child1 = {2: '"948"',
                              3: '"930",',
                              4: '"930",'}
        self.stage2_child2 = {2: '',
                              3: '"475"',
                              4: '"475"'}
        self.stage3_child1 = {2: '"939",',
                              3: '"948"',
                              4: '"939",'}
        self.stage3_child2 = {2: '"484"',
                              3: '',
                              4: '"484"'}
        self.stage5_parent = {2: '"466"',
                              3: '"475"',
                              4: '"484"'}

        # second group
        self.stage6_child1 = {2: '"984"',
                              3: '"966",',
                              4: '"966",'}
        self.stage6_child2 = {2: '',
                              3: '"543"',
                              4: '"543"'}
        self.stage7_child1 = {2: '"975",',
                              3: '"984"',
                              4: '"975",'}
        self.stage7_child2 = {2: '"563"',
                              3: '',
                              4: '"563"'}
        self.stage9_parent = {2: '"523"',
                              3: '"543"',
                              4: '"563"'}

        # third group
        self.stage10_child1 = {2: '"1020"',
                               3: '"1002",',
                               4: '"1002",'}
        self.stage10_child2 = {2: '',
                               3: '"619"',
                               4: '"619"'}
        self.stage11_child1 = {2: '"1011",',
                               3: '"1020"',
                               4: '"1011",'}
        self.stage11_child2 = {2: '"638"',
                               3: '',
                               4: '"638"'}
        self.stage13_parent = {2: '"600"',
                               3: '"619"',
                               4: '"638"'}

        # fourth group
        self.stage14_child1 = {2: '"1056"',
                               3: '"1038",',
                               4: '"1038",'}
        self.stage14_child2 = {2: '',
                               3: '"727"',
                               4: '"727"'}
        self.stage15_child1 = {2: '"1047",',
                               3: '"1056"',
                               4: '"1047",'}
        self.stage15_child2 = {2: '"757"',
                               3: '',
                               4: '"757"'}
        self.stage17_parent = {2: '"697"',
                               3: '"727"',
                               4: '"757"'}

        # fifth group
        self.stage18_child1 = {2: '"1092"',
                               3: '"1074",',
                               4: '"1074",'}
        self.stage18_child2 = {2: '',
                               3: '"846"',
                               4: '"846"'}
        self.stage19_child1 = {2: '"1083",',
                               3: '"1092"',
                               4: '"1083",'}
        self.stage19_child2 = {2: '"876"',
                               3: '',
                               4: '"876"'}
        self.stage21_parent = {2: '"816"',
                               3: '"846"',
                               4: '"876"'}

        self.stage3_blocks = ['930', '469', '933', '472', '936', '475']
        self.stage4_blocks = ['939', '478', '942', '481', '945', '484']
        self.stage7_blocks = ['966', '526', '969', '529', '530', '531', '532', '533', '534', '536', '537', '539', '540', '972', '543']
        self.stage8_blocks = ['975', '546', '978', '549', '550', '551', '552', '553', '554', '556', '557', '559', '560', '981', '563']
        self.stage11_blocks = ['1002', '604', '605', '606', '608', '1005', '612', '613', '614', '616', '1008', '619']
        self.stage12_blocks = ['1011', '623', '624', '625', '627', '1014', '631', '632', '633', '635', '1017', '638']
        self.stage15_blocks = ['1038', '701', '702', '703', '705', '1041', '709', '710', '711', '713', '714', '715', '716', '717', '718', '720', '721', '723', '724', '1044', '727']
        self.stage16_blocks = ['1047', '731', '732', '733', '735', '1050', '739', '740', '741', '743', '744', '745', '746', '747', '748', '750', '751', '753', '754', '1053', '757']
        self.stage19_blocks = ['1074', '820', '821', '822', '824', '1077', '828', '829', '830', '832', '833', '834', '835', '836', '837', '839', '840', '842', '843', '1080', '846']
        self.stage20_blocks = ['1083', '850', '851', '852', '854', '1086', '858', '859', '860', '862', '863', '864', '865', '866', '867', '869', '870', '872', '873', '1089', '876']


    def create_annette_format(self, ks, e, d, r=224):
        start = datetime.datetime.now()

        # remove unneeded blocks
        remove_blocks = []
        if d[0] == 2:
            remove_blocks.extend(self.stage3_blocks)
            remove_blocks.extend(self.stage4_blocks)
        if d[0] == 3:
            remove_blocks.extend(self.stage4_blocks)

        if d[1] == 2:
            remove_blocks.extend(self.stage7_blocks)
            remove_blocks.extend(self.stage8_blocks)
        if d[1] == 3:
            remove_blocks.extend(self.stage8_blocks)

        if d[2] == 2:
            remove_blocks.extend(self.stage11_blocks)
            remove_blocks.extend(self.stage12_blocks)
        if d[2] == 3:
            remove_blocks.extend(self.stage12_blocks)

        if d[3] == 2:
            remove_blocks.extend(self.stage15_blocks)
            remove_blocks.extend(self.stage16_blocks)
        if d[3] == 3:
            remove_blocks.extend(self.stage16_blocks)

        if d[4] == 2:
            remove_blocks.extend(self.stage19_blocks)
            remove_blocks.extend(self.stage20_blocks)
        if d[4] == 3:
            remove_blocks.extend(self.stage20_blocks)

        for block in remove_blocks:
            del self.json_file['layers'][block]

        network_template = json.dumps(self.json_file, indent=4)

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
            '"stage2_child1",': '%s' % (self.stage2_child1[d[0]]),
            '"stage2_child2"': '%s' % (self.stage2_child2[d[0]]),
            '"stage3_child1",': '%s' % (self.stage3_child1[d[0]]),
            '"stage3_child2"': '%s' % (self.stage3_child2[d[0]]),
            '"stage5_parent"': '%s' % (self.stage5_parent[d[0]]),

            '"stage6_child1",': '%s' % (self.stage6_child1[d[1]]),
            '"stage6_child2"': '%s' % (self.stage6_child2[d[1]]),
            '"stage7_child1",': '%s' % (self.stage7_child1[d[1]]),
            '"stage7_child2"': '%s' % (self.stage7_child2[d[1]]),
            '"stage9_parent"': '%s' % (self.stage9_parent[d[1]]),

            '"stage10_child1",': '%s' % (self.stage10_child1[d[2]]),
            '"stage10_child2"': '%s' % (self.stage10_child2[d[2]]),
            '"stage11_child1",': '%s' % (self.stage11_child1[d[2]]),
            '"stage11_child2"': '%s' % (self.stage11_child2[d[2]]),
            '"stage13_parent"': '%s' % (self.stage13_parent[d[2]]),

            '"stage14_child1",': '%s' % (self.stage14_child1[d[3]]),
            '"stage14_child2"': '%s' % (self.stage14_child2[d[3]]),
            '"stage15_child1",': '%s' % (self.stage15_child1[d[3]]),
            '"stage15_child2"': '%s' % (self.stage15_child2[d[3]]),
            '"stage17_parent"': '%s' % (self.stage17_parent[d[3]]),

            '"stage18_child1",': '%s' % (self.stage18_child1[d[4]]),
            '"stage18_child2"': '%s' % (self.stage18_child2[d[4]]),
            '"stage19_child1",': '%s' % (self.stage19_child1[d[4]]),
            '"stage19_child2"': '%s' % (self.stage19_child2[d[4]]),
            '"stage21_parent"': '%s' % (self.stage21_parent[d[4]]),
        }
        replace_patterns = dict((re.escape(k), v) for k, v in replace_pattern.items())
        pattern = re.compile(("|".join(replace_patterns.keys())))
        # replace fields with actual values
        annette_json = pattern.sub(lambda m: replace_patterns[re.escape(m.group(0))], network_template)
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
    ofa_net = OFAMobileNetV3(ks_list=[3, 5, 7], depth_list=[2, 3, 4], expand_ratio_list=[3, 4, 6])
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


def rename_nodes(path_json_net):
    replace_pattern = {}

    json_file = json.load(open(path_json_net))
    for new_name, element in enumerate(json_file['layers']):
        if element != 'input.1':
            replace_pattern['"' + element + '"'] = str(new_name)

    replace_patterns = dict((re.escape(k), v) for k, v in replace_pattern.items())
    pattern = re.compile(("|".join(replace_patterns.keys())))
    # replace fields with actual values
    json_str = json.dumps(json_file, indent=4)
    renamed_nodes = pattern.sub(lambda m: replace_patterns[re.escape(m.group(0))], json_str)

    # save file
    new_file_name = os.path.splitext(path_json_net)[0] + '_reordered.json'
    new_json_file = open(new_file_name, 'w')
    new_json_file.write(renamed_nodes)
    new_json_file.close()
    return new_file_name



#########################

sub_net_architectures = [
    # {'ks': [3, 3, 7, 7, 3, 7, 3, 3, 3, 7, 3, 7, 7, 3, 3, 5, 3, 5, 3, 7], 'e': [3, 4, 6, 4, 4, 3, 4, 4, 6, 4, 3, 4, 3, 3, 4, 6, 4, 6, 6, 4], 'd': [2, 2, 2, 2, 3], 'r': 224}
    # {'ks': [5, 7, 5, 3, 7, 5, 3, 3, 5, 7, 5, 7, 7, 7, 3, 7, 7, 7, 7, 5], 'e': [3, 3, 4, 6, 3, 4, 4, 6, 3, 6, 6, 4, 4, 3, 6, 6, 6, 6, 4, 3], 'd': [4, 4, 4, 4, 4], 'r': 192}
    # {'ks': [3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7], 'e': [4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], 'd': [3, 4, 4, 4, 4], 'r': 224},
    # {'ks': [7, 3, 3, 7, 5, 5, 7, 5, 3, 3, 3, 5, 5, 3, 7, 7, 7, 3, 7, 3], 'e': [3, 6, 4, 3, 3, 3, 4, 3, 6, 3, 4, 3, 4, 6, 3, 4, 4, 4, 6, 3], 'd': [4, 4, 4, 4, 4], 'r': 128},
    # {'ks': [5, 7, 3, 3, 5, 5, 3, 3, 3, 5, 3, 3, 3, 7, 5, 7, 3, 3, 7, 5], 'e': [4, 3, 6, 6, 6, 3, 6, 4, 4, 6, 6, 6, 4, 6, 6, 6, 4, 6, 6, 3], 'd': [4, 4, 4, 4, 4], 'r': 160},
    # {'ks': [5, 5, 7, 5, 7, 5, 3, 7, 3, 3, 7, 7, 3, 3, 3, 5, 5, 7, 3, 7], 'e': [6, 3, 3, 6, 3, 3, 6, 6, 6, 3, 6, 4, 4, 4, 4, 4, 6, 3, 4, 6], 'd': [4, 4, 4, 4, 4], 'r': 160},
    # {'ks': [3, 5, 7, 3, 3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 5, 3, 7, 5, 7, 5], 'e': [3, 4, 6, 4, 6, 6, 4, 4, 4, 3, 3, 4, 6, 4, 6, 6, 4, 4, 4, 6], 'd': [4, 4, 4, 4, 4], 'r': 192},
]

logfile_name = 'test_results_converter.txt'
logfile = open(logfile_name, 'w')
error_file_name = 'errors_converter.txt'
error_file = open(error_file_name, 'w')

# for sub_net_arch in sub_net_architectures:
for i in range(10):
    file_name_annette, sub_net_arch = ofa_2_onnx_2_annette()
    # file_name_annette, _ = ofa_2_onnx_2_annette(sub_net_arch)

    logfile.write('***Test***\n')
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

    file1 = rename_nodes(file_name_annette)
    file2 = rename_nodes(file_name_converted_annette)


    # match = compare_files(file_name_annette, file_name_converted_annette)
    match = compare_files(file1, file2)

    if match:
        logfile.write('\nConversion is correct :)\n')
        print('\nConversion is correct :)\n')
    else:
        logfile.write('\n>>>>>> !!! ERROR !!! >>>>>>\n')
        error_file.write(str(sub_net_arch))
        print('\n>>>>>> !!! ERROR !!! >>>>>>\n')

print('~~~* fin *~~~')
