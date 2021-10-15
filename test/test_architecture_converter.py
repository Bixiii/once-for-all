import unittest
import random
import json
import re
import os
from pathlib import Path

import onnx
from onnxsim import simplify

from annette.graph import ONNXGraph

from generate_annette_format import AnnetteConverter
from utils import export_as_onnx, architecture_config_2_str
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3


# compare tow files line by line, returns number of lines that are not equal
def compare_files(file_name_1, file_name_2):
    with open(file_name_1) as f1, open(file_name_2) as f2:
        lines1 = (line.rstrip('\n') for line in f1)
        lines2 = (line.rstrip('\n') for line in f2)

        num_diff_lines = 0
        for line1, line2 in zip(lines1, lines2):
            if line1 != line2:
                num_diff_lines += 1
        f1.close()
        f2.close()

    return num_diff_lines


def ofa_2_onnx_2_annette(sub_net_arch=None, keep_files=False):
    '''
    Convert OFA architecture description (MobileNetV3) to ONNX and then to ANNETTE format

    Args:
        sub_net_arch (): OFA architecture description, if None a random configuration will be sampled

    Returns: path to the file containing the ANNETTE format, subnet architecture

    '''
    image_size_list = [128, 160, 192, 224]
    ofa_net = OFAMobileNetV3(ks_list=[3, 5, 7], depth_list=[2, 3, 4], expand_ratio_list=[3, 4, 6])
    if sub_net_arch is None:
        sub_net_arch = ofa_net.sample_active_subnet()
        sub_net_arch['r'] = random.choice(image_size_list)
    ofa_net.set_active_subnet(ks=sub_net_arch['ks'], e=sub_net_arch['e'], d=sub_net_arch['d'])
    sub_net = ofa_net.get_active_subnet()

    config_str = ''
    if sub_net_arch is not None:
        config_str = architecture_config_2_str(sub_net_arch)

    file_name = r'./out_data/mbv3_%s.onnx' % config_str
    file_name_simplified = r'./out_data/mbv3_simplified_%s.onnx' % config_str
    file_name_annette = r'./out_data/%s-mbv3.json' % config_str

    export_as_onnx(sub_net, file_name, sub_net_arch['r'])
    onnx_model = onnx.load(file_name)
    simplified, check = simplify(onnx_model)
    onnx.save(simplified, file_name_simplified)
    onnx_model_simplified = onnx.load(file_name_simplified)

    onnx_network = ONNXGraph(file_name_simplified)
    annette_graph = onnx_network.onnx_to_annette(file_name_simplified, ['input.1'], verbose=False)
    annette_graph.to_json(Path(file_name_annette))

    fd = open(file_name_annette)
    annette_graph_json = json.dumps(json.load(fd), indent=4)
    fd.close()
    if not keep_files:
        os.remove(file_name)
        os.remove(file_name_simplified)
    return file_name_annette, sub_net_arch


def rename_nodes(path_json_net):
    """
    Rename elements in ANNETTE json format
    Args:
        path_json_net (): path to json file containing network in ANNETTE format

    Returns: path to file with renamed nodes
    """

    fd = open(path_json_net)
    json_file = json.load(fd)
    fd.close()
    input_layers = json_file['input_layers']
    output_layers = json_file['output_layers']

    # get new name for each node
    replace_pattern = {}
    for new_name, element in enumerate(json_file['layers']):
        if element in input_layers:
            continue
        replace_pattern['"' + element + '"'] = '"' + str(new_name) + '"'

    replace_patterns = dict((re.escape(k), v) for k, v in replace_pattern.items())
    pattern = re.compile(("|".join(replace_patterns.keys())))

    # replace values
    json_str = json.dumps(json_file, indent=4)
    renamed_nodes = pattern.sub(lambda m: replace_patterns[re.escape(m.group(0))], json_str)

    # save file
    new_file_name = os.path.splitext(path_json_net)[0] + '_reordered.json'
    new_json_file = open(new_file_name, 'w')
    new_json_file.write(renamed_nodes)
    new_json_file.close()
    return new_file_name


# TODO check why mult is not causing error (it should)
def sanity_check(path_annette_net):
    '''
    Check if each node (except input and output) has at least one child and parent
    Args:
        path_annette_net (): path to ANNETTE json file

    '''
    print("Sanity Check")
    json_file = json.load(open(path_annette_net))
    input_layers = json_file['input_layers']
    output_layers = json_file['output_layers']

    for element in json_file['layers']:
        if element in input_layers or element in output_layers:
            continue  # skip input and output

        # check if each node has a parent and a child (everything is connected)
        try:
            assert (len(json_file['layers'][element]['children']) > 0)
        except AssertionError:
            print('Node %s has no children' % element)
        try:
            assert (len(json_file['layers'][element]['parents']) > 0)
        except AssertionError:
            print('Node %s has no parents' % element)

        success = True
        # check if input and output shapes match
        for child in json_file['layers'][element]['children']:
            try:
                assert (json_file['layers'][element]['output_shape'] == json_file['layers'][child]['input_shape'])
            except AssertionError as e:
                success = False
                print('Output of parent "%s" (%s) %s does not match input of child "%s" (%s) %s' %
                      (element, str(json_file['layers'][element]['type']),
                       str(json_file['layers'][element]['output_shape']), child,
                       str(json_file['layers'][child]['type']), str(json_file['layers'][child]['input_shape'])))

        return success


class TestConverter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # set up converter
        json_network_template_path = r'resources\generalized_mbv3.json'
        cls.converter = AnnetteConverter(json_network_template_path)

    @classmethod
    def tearDownClass(cls):
        cls.converter = None

    def test_converter(self):
        sub_net_architectures = [
            {'ks': [3, 5, 7, 3, 3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 5, 3, 7, 5, 7, 5],
             'e': [3, 4, 6, 4, 6, 6, 4, 4, 4, 3, 3, 4, 6, 4, 6, 6, 4, 4, 4, 6], 'd': [4, 2, 2, 3, 4], 'r': 192},
        ]
        for sub_net_arch in sub_net_architectures:
            mbv3_annette = self.converter.create_annette_format(**sub_net_arch)
            tmp_file = './out_data/mbv3-annette_%s.json' % architecture_config_2_str(sub_net_arch)
            new_json_file = open(tmp_file, 'w')
            new_json_file.write(mbv3_annette)
            new_json_file.close()

            self.assertTrue(sanity_check(tmp_file))
            os.remove(tmp_file)



        print('hi')

    def test_full_conversion(self, keep_files=True):

        error_file_name = './out_data/errors_converter.txt'
        error_file = open(error_file_name, 'w')

        # sub_net_architectures = [
        #     {'ks': [3, 5, 7, 3, 3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 5, 3, 7, 5, 7, 5],
        #      'e': [3, 4, 6, 4, 6, 6, 4, 4, 4, 3, 3, 4, 6, 4, 6, 6, 4, 4, 4, 6], 'd': [4, 2, 2, 3, 4], 'r': 192},
        #     # {'ks': [3, 5, 7, 3, 3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 5, 3, 7, 5, 7, 5],
        #     #  'e': [3, 4, 6, 4, 6, 6, 4, 4, 4, 3, 3, 4, 6, 4, 6, 6, 4, 4, 4, 6], 'd': [4, 2, 2, 3, 4], 'r': 192},
        # ]
        # for sub_net_arch in sub_net_architectures:
        #     # create reference
        #     file_name_annette, _ = ofa_2_onnx_2_annette(sub_net_arch)

        for i in range(3):
            # create reference
            file_name_annette, sub_net_arch = ofa_2_onnx_2_annette()

            # do conversion
            annette_json = self.converter.create_annette_format(**sub_net_arch)

            # save file
            file_name_converted_annette = './out_data/mbv3-annette_%s.json' % (architecture_config_2_str(sub_net_arch))
            new_json_file = open(file_name_converted_annette, 'w')
            new_json_file.write(annette_json)
            new_json_file.close()

            file1 = rename_nodes(file_name_annette)
            file2 = rename_nodes(file_name_converted_annette)

            match = compare_files(file1, file2)

            if not keep_files:
                os.remove(file1)
                os.remove(file2)
                os.remove(file_name_annette)
                os.remove(file_name_converted_annette)

            # first line contains file name and will be different, all other liens should match
            if match != 1:
                error_file.write('Error when converting %s\n' % (architecture_config_2_str(sub_net_arch)))
            self.assertEqual(match, 1, ('Error when converting %s' % (architecture_config_2_str(sub_net_arch))))

        error_file.close()
