import datetime
import annette.graph as graph

from annette import get_database
from annette.estimation.layer_model import Layer_model
from annette.estimation.mapping_model import Mapping_model
from annette.graph import AnnetteGraph
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3, OFAResNets
from result_net_configs import *
from utils import timestamp_string, export_as_onnx, architecture_config_2_str
import onnx
from onnxsim import simplify


class AnnetteEstimator:

    def __init__(self, mapping, layers):
        valid_mapping = ['dnndk', 'ov', 'ov2', 'ov3']
        valid_layers = ['dnndk-mixed',
                        'dnndk-ref_roofline',
                        'dnndk-roofline',
                        'dnndk-statistical',
                        'ncs2-mixed',
                        'ncs2-ref_roofline',
                        'ncs2-roofline',
                        'ncs2-statistical',
                        ]

        assert mapping in valid_mapping
        assert layers in valid_layers

        self.mapping = mapping
        self.layers = layers

        # load models
        self.opt = Mapping_model.from_json(get_database('models', 'mapping', self.mapping + '.json'))
        self.mod = Layer_model.from_json(get_database('models', 'layer', self.layers + '.json'))

    def estimate(self, file_path, net_name='net'):

        # load from annette json file
        model = AnnetteGraph(net_name, file_path)

        # estimate
        self.opt.run_optimization(model)
        res = self.mod.estimate_model(model)

        return res[0]


def onnx_to_annette(onnx_network_path, inputs=None, annette_file_path='annette_tmp_file.json'):
    onnx_network = graph.ONNXGraph(onnx_network_path)
    annette_graph = onnx_network.onnx_to_annette(onnx_network_path, inputs)
    annette_graph.to_json(annette_file_path)
    return annette_file_path


def ofa_config_to_onnx(ofa_net_config, net_architecture_name):
    if net_architecture_name == 'MobileNetV3':
        ks_size_list = [3, 5, 7]
        depth_list = [2, 3, 4]
        expand_list = [3, 4, 6]
        ofa_network = OFAMobileNetV3(
            n_classes=1000,
            dropout_rate=0,
            ks_list=ks_size_list,
            depth_list=depth_list,
            expand_ratio_list=expand_list,
        )
        subnet = ofa_network.get_active_subnet(
            ofa_network.set_active_subnet(ks=ofa_net_config['ks'], d=ofa_net_config['d'], e=ofa_net_config['e']))
    elif net_architecture_name == 'ResNet50':
        depth_list = [0, 1, 2]
        expand_list = [0.2, 0.25, 0.35]
        width_mult_list = [0.65, 0.8, 1.0]
        ofa_network = OFAResNets(
            n_classes=1000,
            dropout_rate=0,
            depth_list=depth_list,
            expand_ratio_list=expand_list,
            width_mult_list=width_mult_list,
            small_input_stem=False,
            dataset='imagenet'
        )
        subnet = ofa_network.get_active_subnet(
            ofa_network.set_active_subnet(d=ofa_net_config['d'], e=ofa_net_config['e'], w=ofa_net_config['w']))
    else:
        raise NotImplementedError

    model_file_name = 'logs/' + timestamp_string() + '.onnx'
    simplified_model_file_name = 'logs/' + net_architecture_name + '_' + architecture_config_2_str(ofa_net_config) + '_simplified.onnx'
    export_as_onnx(subnet, model_file_name)
    onnx_model = onnx.load(model_file_name)
    simplified_model, check = simplify(onnx_model)
    onnx.save(simplified_model, simplified_model_file_name)
    print('Saved to <', simplified_model_file_name, '>')
    return simplified_model_file_name


def main():

    file = open('new_annette_flop_ncs2_estimates.txt', 'w')
    for config in resnet50_flop_constrained:

        onnx_file = ofa_config_to_onnx(config, 'ResNet50')
        annette_file = onnx_file[:-5] + '_ov2.json'

        annette_network_path = onnx_to_annette(onnx_file, inputs='input.1', annette_file_path=annette_file)

        annette_estimator = AnnetteEstimator(mapping='dnndk', layers='dnndk-mixed')
        start = datetime.datetime.now()
        # latency = annette_estimator.estimate(file_path, 'mbv3')
        latency = annette_estimator.estimate(annette_network_path, 'mbv3')
        end = datetime.datetime.now()
        file.write('Estimated latency is %.5f (estimation took %.0f ms)' % (latency, (end - start).total_seconds() * 1000) + '\n')
        print('Estimated latency is %.5f (estimation took %.0f ms)' % (latency, (end - start).total_seconds() * 1000))
    file.close()


if __name__ == '__main__':
    main()
