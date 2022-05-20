# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy

from utils import *

from .latency_lookup_table import *

from generate_annette_format import AnnetteConverter
from annette import get_database
from annette.estimation.layer_model import Layer_model
from annette.estimation.mapping_model import Mapping_model
from annette.graph import AnnetteGraph, ONNXGraph
from pathlib import Path
import onnx
from onnxsim import simplify
from ofa.utils.layers import LinearLayer


class BaseEfficiencyModel:

    def __init__(self, ofa_net):
        self.ofa_net = ofa_net

    def get_active_subnet_config(self, arch_dict):
        arch_dict = copy.deepcopy(arch_dict)
        image_size = arch_dict.pop('image_size')
        self.ofa_net.set_active_subnet(**arch_dict)
        active_net_config = self.ofa_net.get_active_net_config()
        return active_net_config, image_size

    def get_efficiency(self, arch_dict):
        raise NotImplementedError


class ProxylessNASFLOPsModel(BaseEfficiencyModel):

    def get_efficiency(self, arch_dict):
        active_net_config, image_size = self.get_active_subnet_config(arch_dict)
        return ProxylessNASLatencyTable.count_flops_given_config(active_net_config, image_size)


class Mbv3FLOPsModel(BaseEfficiencyModel):

    def get_efficiency(self, arch_dict):
        active_net_config, image_size = self.get_active_subnet_config(arch_dict)
        return MBv3LatencyTable.count_flops_given_config(active_net_config, image_size)

    def predict_efficiency(self, arch_dict):
        return self.get_efficiency(arch_dict)


class ResNet50FLOPsModel(BaseEfficiencyModel):

    def get_efficiency(self, arch_dict):
        active_net_config, image_size = self.get_active_subnet_config(arch_dict)
        return ResNet50LatencyTable.count_flops_given_config(active_net_config, image_size)


class ResNet50AnnetteLUT(BaseEfficiencyModel):

    def __init__(self, ofa_net, loaded_lut):
        self.ofa_net = ofa_net
        self.loaded_lut = loaded_lut

    def get_efficiency(self, arch_dict):
        active_net_config, image_size = self.get_active_subnet_config(arch_dict)
        arch_dict['image_size'] = image_size
        latency, _ = self.ofa_net.predict_with_annette_lut(loaded_lut=self.loaded_lut, subnet_config=arch_dict)
        return latency


class ProxylessNASLatencyModel(BaseEfficiencyModel):

    def __init__(self, ofa_net, lookup_table_path_dict):
        super(ProxylessNASLatencyModel, self).__init__(ofa_net)
        self.latency_tables = {}
        for image_size, path in lookup_table_path_dict.items():
            self.latency_tables[image_size] = ProxylessNASLatencyTable(
                local_dir='/tmp/.ofa_latency_tools/', url=os.path.join(path, '%d_lookup_table.yaml' % image_size))

    def get_efficiency(self, arch_dict):
        active_net_config, image_size = self.get_active_subnet_config(arch_dict)
        return self.latency_tables[image_size].predict_network_latency_given_config(active_net_config, image_size)


class Mbv3LatencyModel(BaseEfficiencyModel):

    def __init__(self, ofa_net, lookup_table_path_dict):
        super(Mbv3LatencyModel, self).__init__(ofa_net)
        self.latency_tables = {}
        for image_size, path in lookup_table_path_dict.items():
            self.latency_tables[image_size] = MBv3LatencyTable(
                local_dir='/tmp/.ofa_latency_tools/', url=os.path.join(path, '%d_lookup_table.yaml' % image_size))

    def get_efficiency(self, arch_dict):
        active_net_config, image_size = self.get_active_subnet_config(arch_dict)
        return self.latency_tables[image_size].predict_network_latency_given_config(active_net_config, image_size)


class AnnetteLatencyModel(BaseEfficiencyModel):

    mappings = ['dnndk', 'ov']  # dnndk <- Xlinx ZCU 10, 2ov <- open vino
    layers = ['dnndk-mixed',
              'dnndk-ref_roofline',
              'dnndk-roofline',
              'dnndk-statistical',  # statistical <- random forest
              'ncs2-mixed',  # ncs2 <- intel neural compute stick
              'ncs2-ref_roofline',
              'ncs2-roofline',
              'ncs2-statistical',
              ]

    def __init__(self, ofa_net, model='dnndk-mixed'):
        super().__init__(ofa_net)
        os.makedirs('./exp/tmp/', exist_ok=True)

        assert(model in AnnetteLatencyModel.layers)
        layer = model

        if 'dnndk' in layer:
            mapping = 'dnndk'
        elif 'ncs2' in layer:
            mapping = 'ov'
        else:
            raise NotImplementedError

        # load models
        self.opt = Mapping_model.from_json(get_database('models', 'mapping', mapping + '.json'))
        self.mod = Layer_model.from_json(get_database('models', 'layer', layer + '.json'))

        logger.info('Initialized Annette with layer model <%s> and mapping model <%s>' % (layer, mapping))

        # set up ofa-2-annette converter
        network_template = project_root() + '/resources/generalized_mbv3.json'
        self.annette_converter = AnnetteConverter(network_template)

    def predict_efficiency(self, arch_dict):
        # logger.debug('Start ANNETTE efficiency prediction')
        annette_network = self.annette_converter.create_annette_format(ks=arch_dict['ks'], e=arch_dict['e'],
                                                                       d=arch_dict['d'], r=arch_dict['image_size'])
        # save file
        annette_network_path = './exp/tmp/mbv3-annette_%s.json' % (architecture_config_2_str(arch_dict))
        new_json_file = open(annette_network_path, 'w')
        new_json_file.write(annette_network)
        new_json_file.close()

        # load from annette json file
        model = AnnetteGraph('ofa-net', annette_network_path)

        # estimate
        self.opt.run_optimization(model)
        res = self.mod.estimate_model(model)

        # logger.debug('Finished ANNETTE efficiency prediction')
        return res[0]

    def get_efficiency(self, arch_dict):
        return self.predict_efficiency(arch_dict)


class AnnetteLatencyModelResNet50(BaseEfficiencyModel):

    mappings = ['dnndk', 'ov']  # dnndk <- Xlinx ZCU 10, 2ov <- open vino
    layers = ['dnndk-mixed',
              'dnndk-ref_roofline',
              'dnndk-roofline',
              'dnndk-statistical',  # statistical <- random forest
              'ncs2-mixed',  # ncs2 <- intel neural compute stick
              'ncs2-ref_roofline',
              'ncs2-roofline',
              'ncs2-statistical',
              ]

    def __init__(self, ofa_net, model='dnndk-mixed'):
        super().__init__(ofa_net)
        os.makedirs('./tmp/', exist_ok=True)

        assert(model in AnnetteLatencyModel.layers)
        layer = model

        if 'dnndk' in layer:
            mapping = 'dnndk'
        elif 'ncs2' in layer:
            mapping = 'ov2'
        else:
            raise NotImplementedError

        # load models
        self.opt = Mapping_model.from_json(get_database('models', 'mapping', mapping + '.json'))
        self.mod = Layer_model.from_json(get_database('models', 'layer', layer + '.json'))

        logger.info('Initialized Annette with layer model <%s> and mapping model <%s>' % (layer, mapping))

    def predict_efficiency(self, arch_dict):

        logger.debug('Start ANNETTE efficiency prediction')
        # get ANNETTE latency estimation: export as ONNX, load ONNX for ANNETTE, make prediction
        subnet = self.ofa_net.get_active_subnet(
            self.ofa_net.set_active_subnet(d=arch_dict['d'], e=arch_dict['e'], w=arch_dict['w']))

        model_file_name = './tmp/' + timestamp_string() + '.onnx'
        simplified_model_file_name = './tmp/' + timestamp_string() + 'simplified.onnx'
        annette_model_file_name = './tmp/' + timestamp_string() + '.json'
        export_as_onnx(subnet, model_file_name)

        onnx_model = onnx.load(model_file_name)
        simplified_model, check = simplify(onnx_model)
        onnx.save(simplified_model, simplified_model_file_name)

        onnx_network = ONNXGraph(simplified_model_file_name)
        annette_graph = onnx_network.onnx_to_annette(simplified_model_file_name, ['input.1'], name_policy='renumerate')
        json_file = Path(annette_model_file_name)
        annette_graph.to_json(json_file)

        model = AnnetteGraph('ofa-net', annette_model_file_name)

        self.opt.run_optimization(model)
        res = self.mod.estimate_model(model)

        # logger.info('Finished ANNETTE efficiency prediction, result <' + str(res[0]) + '>')

        os.remove(model_file_name)
        os.remove(simplified_model_file_name)
        os.remove(annette_model_file_name)
        return res[0]

    def get_efficiency(self, arch_dict):
        return self.predict_efficiency(arch_dict)


class AnnetteLatencyLayerPrediction:

    mappings = ['dnndk', 'ov']  # dnndk <- Xlinx ZCU 10, 2ov <- open vino
    layers = ['dnndk-mixed',
              'dnndk-ref_roofline',
              'dnndk-roofline',
              'dnndk-statistical',  # statistical <- random forest
              'ncs2-mixed',  # ncs2 <- intel neural compute stick
              'ncs2-ref_roofline',
              'ncs2-roofline',
              'ncs2-statistical',
              ]

    def __init__(self, model='dnndk-mixed'):
        os.makedirs('./tmp/', exist_ok=True)

        assert(model in AnnetteLatencyModel.layers)
        layer = model

        if 'dnndk' in layer:
            mapping = 'dnndk'
        elif 'ncs2' in layer:
            mapping = 'ov2'
        else:
            raise NotImplementedError

        # load models
        self.opt = Mapping_model.from_json(get_database('models', 'mapping', mapping + '.json'))
        self.mod = Layer_model.from_json(get_database('models', 'layer', layer + '.json'))

        logger.info('Initialized Annette with layer model <%s> and mapping model <%s>' % (layer, mapping))
        self.debug_file = open('./tmp/debug_prints.txt', 'w')

    def predict_efficiency(self, network_layer, input_size):

        logger.debug('Start ANNETTE efficiency prediction')
        # get ANNETTE latency estimation: export as ONNX, load ONNX for ANNETTE, make prediction

        model_file_name = './tmp/' + timestamp_string() + '.onnx'
        simplified_model_file_name = './tmp/' + timestamp_string() + 'simplified.onnx'
        annette_model_file_name = './tmp/' + timestamp_string() + '.json'
        export_layer_as_onnx(network_layer, model_file_name, input_size)

        onnx_model = onnx.load(model_file_name)
        simplified_model, check = simplify(onnx_model)
        onnx.save(simplified_model, simplified_model_file_name)

        onnx_network = ONNXGraph(simplified_model_file_name)
        # if isinstance(network_layer, LinearLayer):
        #     annette_graph = onnx_network.onnx_to_annette(simplified_model_file_name, ['input'],
        #                                                  name_policy='renumerate')
        # else:
        #     annette_graph = onnx_network.onnx_to_annette(simplified_model_file_name, ['input.1'],
        #                                                  name_policy='renumerate')
        annette_graph = onnx_network.onnx_to_annette(simplified_model_file_name, None,
                                                     name_policy='renumerate')
        json_file = Path(annette_model_file_name)
        annette_graph.to_json(json_file)

        model = AnnetteGraph('ofa-net', annette_model_file_name)

        self.opt.run_optimization(model)
        res = self.mod.estimate_model(model)

        # logger.info('Finished ANNETTE efficiency prediction, result <' + str(res[0]) + '>')

        os.remove(model_file_name)
        os.remove(simplified_model_file_name)
        os.remove(annette_model_file_name)
        return res[0]

    def get_efficiency(self, arch_dict):
        raise NotImplementedError
        # return self.predict_efficiency(arch_dict)