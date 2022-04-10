import random

from torch import nn

from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import DynamicConvLayer, DynamicLinearLayer
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import DynamicResNetBottleneckBlock
from ofa.imagenet_classification.networks.resnets import InputStemLayers, ResNetStageLayers, ResNetBlockLayer, ResNetClassifier
from ofa.nas.efficiency_predictor import AnnetteLatencyLayerPrediction
from ofa.utils.layers import IdentityLayer, ResidualBlock
from ofa.imagenet_classification.networks import ResNets
from ofa.utils import make_divisible, val2list, MyNetwork
from utils import export_layer_as_onnx
import pickle

__all__ = ['OFAResNets']


class OFAResNets(ResNets):

    def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0,
                 depth_list=2, expand_ratio_list=0.25, width_mult_list=1.0, small_input_stem=False, dataset='imagenet'):

        self.depth_list = val2list(depth_list)
        self.expand_ratio_list = val2list(expand_ratio_list)
        self.width_mult_list = val2list(width_mult_list)
        self.small_input_stem = small_input_stem
        # sort
        self.depth_list.sort()
        self.expand_ratio_list.sort()
        self.width_mult_list.sort()
        self.dataset = dataset

        self.logfile = open('tmp/logging.log', 'w+')

        input_channel = [
            make_divisible(64 * width_mult, MyNetwork.CHANNEL_DIVISIBLE) for width_mult in self.width_mult_list
        ]
        mid_input_channel = [
            make_divisible(channel // 2, MyNetwork.CHANNEL_DIVISIBLE) for channel in input_channel
        ]

        stage_width_list = ResNets.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = [
                make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE) for width_mult in self.width_mult_list
            ]

        n_block_list = [base_depth + max(self.depth_list) for base_depth in ResNets.BASE_DEPTH_LIST]
        stride_list = [1, 2, 2, 2]

        if dataset == 'imagenet':
            input_stem_kernel_size = 7
            input_stem_stride = 2
            max_pooling = True
            downsample_mode = 'avgpool_conv'
        elif dataset == 'cifar10':
            input_stem_kernel_size = 3
            input_stem_stride = 1
            max_pooling = False
            downsample_mode = 'conv'

        # build input stem
        if small_input_stem:
            input_stem = [
                DynamicConvLayer(val2list(3), input_channel, input_stem_kernel_size, stride=input_stem_stride, use_bn=True, act_func='relu'),
            ]
            downsample_mode = 'conv'
        else:
            input_stem = [
                DynamicConvLayer(val2list(3), mid_input_channel, 3, stride=2, use_bn=True, act_func='relu'),
                ResidualBlock(
                    DynamicConvLayer(mid_input_channel, mid_input_channel, 3, stride=1, use_bn=True, act_func='relu'),
                    IdentityLayer(mid_input_channel, mid_input_channel)
                ),
                DynamicConvLayer(mid_input_channel, input_channel, 3, stride=1, use_bn=True, act_func='relu')
            ]

        # blocks
        blocks = []
        for d, width, s in zip(n_block_list, stage_width_list, stride_list):
            for i in range(d):
                stride = s if i == 0 else 1
                bottleneck_block = DynamicResNetBottleneckBlock(
                    input_channel, width, expand_ratio_list=self.expand_ratio_list,
                    kernel_size=3, stride=stride, act_func='relu', downsample_mode=downsample_mode,
                )
                blocks.append(bottleneck_block)
                input_channel = width
        # classifier
        classifier = DynamicLinearLayer(input_channel, n_classes, dropout_rate=dropout_rate)

        super(OFAResNets, self).__init__(input_stem, blocks, classifier, max_pooling=max_pooling)

        # set bn param
        self.set_bn_param(*bn_param)

        # runtime_depth
        self.input_stem_skipping = 0
        self.runtime_depth = [0] * len(n_block_list)

    @property
    def ks_list(self):
        return [3]

    @staticmethod
    def name():
        return 'OFAResNets'

    def forward(self, x):
        for layer in self.input_stem:
            if self.input_stem_skipping > 0 \
                    and isinstance(layer, ResidualBlock) and isinstance(layer.shortcut, IdentityLayer):
                pass
            else:
                x = layer(x)
        if self.max_pooling:
            x = self.max_pooling(x)
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[:len(block_idx) - depth_param]
            for idx in active_idx:
                x = self.blocks[idx](x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ''
        for layer in self.input_stem:
            if self.input_stem_skipping > 0 \
                    and isinstance(layer, ResidualBlock) and isinstance(layer.shortcut, IdentityLayer):
                pass
            else:
                _str += layer.module_str + '\n'
        if self.max_pooling:
            _str += 'max_pooling(ks=3, stride=2)\n'
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[:len(block_idx) - depth_param]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'
        _str += self.global_avg_pool.__repr__() + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': OFAResNets.__name__,
            'bn': self.get_bn_param(),
            'input_stem': [
                layer.config for layer in self.input_stem
            ],
            'blocks': [
                block.config for block in self.blocks
            ],
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            new_key = key
            if new_key in model_dict:
                pass
            elif '.linear.' in new_key:
                new_key = new_key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in new_key:
                new_key = new_key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in new_key:
                new_key = new_key.replace('conv.weight', 'conv.conv.weight')
            else:
                raise ValueError(new_key)
            assert new_key in model_dict, '%s' % new_key
            model_dict[new_key] = state_dict[key]
        super(OFAResNets, self).load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_max_net(self):
        self.set_active_subnet(d=max(self.depth_list), e=max(self.expand_ratio_list), w=len(self.width_mult_list) - 1)

    def set_active_subnet(self, d=None, e=None, w=None, **kwargs):
        depth = val2list(d, len(ResNets.BASE_DEPTH_LIST) + 1)  # +1 for input stem
        expand_ratio = val2list(e, len(self.blocks))
        size_input_stem = 1 if self.small_input_stem else 2
        width_mult = val2list(w, len(ResNets.BASE_DEPTH_LIST) + size_input_stem)

        for block, e in zip(self.blocks, expand_ratio):
            if e is not None:
                block.active_expand_ratio = e

        if not self.small_input_stem:
            if width_mult[0] is not None:
                self.input_stem[1].conv.active_out_channel = self.input_stem[0].active_out_channel = \
                    self.input_stem[0].out_channel_list[width_mult[0]]
            if width_mult[1] is not None:
                self.input_stem[2].active_out_channel = self.input_stem[2].out_channel_list[width_mult[1]]

        if depth[0] is not None:
            self.input_stem_skipping = (depth[0] != max(self.depth_list))
        for stage_id, (block_idx, d, w) in enumerate(zip(self.grouped_block_index, depth[1:], width_mult[2:])):
            if d is not None:
                self.runtime_depth[stage_id] = max(self.depth_list) - d
            if w is not None:
                for idx in block_idx:
                    self.blocks[idx].active_out_channel = self.blocks[idx].out_channel_list[w]

    def sample_active_subnet(self):
        # sample expand ratio
        expand_setting = []
        for block in self.blocks:
            expand_setting.append(random.choice(block.expand_ratio_list))

        # sample depth
        depth_setting = [random.choice([max(self.depth_list), min(self.depth_list)])]
        for stage_id in range(len(ResNets.BASE_DEPTH_LIST)):
            depth_setting.append(random.choice(self.depth_list))

        # sample width_mult
        if not self.small_input_stem:
            width_mult_setting = [
                random.choice(list(range(len(self.input_stem[0].out_channel_list)))),
                random.choice(list(range(len(self.input_stem[2].out_channel_list)))),
            ]
        else:
            width_mult_setting = [
                random.choice(list(range(len(self.input_stem[0].out_channel_list)))),
            ]

        for stage_id, block_idx in enumerate(self.grouped_block_index):
            stage_first_block = self.blocks[block_idx[0]]
            width_mult_setting.append(
                random.choice(list(range(len(stage_first_block.out_channel_list))))
            )

        arch_config = {
            'd': depth_setting,
            'e': expand_setting,
            'w': width_mult_setting
        }
        self.set_active_subnet(**arch_config)
        return arch_config

    def get_active_subnet(self, preserve_weight=True):
        input_stem = [self.input_stem[0].get_active_subnet(3, preserve_weight)]
        if self.input_stem_skipping <= 0:
            input_stem.append(ResidualBlock(
                self.input_stem[1].conv.get_active_subnet(self.input_stem[0].active_out_channel, preserve_weight),
                IdentityLayer(self.input_stem[0].active_out_channel, self.input_stem[0].active_out_channel)
            ))
        input_stem.append(self.input_stem[2].get_active_subnet(self.input_stem[0].active_out_channel, preserve_weight))
        input_channel = self.input_stem[2].active_out_channel

        blocks = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[:len(block_idx) - depth_param]
            for idx in active_idx:
                blocks.append(self.blocks[idx].get_active_subnet(input_channel, preserve_weight))
                input_channel = self.blocks[idx].active_out_channel
        classifier = self.classifier.get_active_subnet(input_channel, preserve_weight)
        subnet = ResNets(input_stem, blocks, classifier, max_pooling=self.max_pooling)

        subnet.set_bn_param(**self.get_bn_param())
        return subnet

    def predict_with_annette_lut(self, loaded_lut, subnet_config, verify=False):
        """
        Estimate the annette latency prediction with a look up table
        Also see: build_annette_lut()

        Args:
            loaded_lut (): path with the saved look-up-table for annette lantency estimations
            subnet_config (): ofa config for subnet
            verify (): if true, additionally subnet from given ofa subnet configuration is returned

        Returns: latency prediction, subnet from ofa subnetwork configuration

        """

        self.set_active_subnet(d=subnet_config['d'], e=subnet_config['e'], w=subnet_config['w'])

        latency = 0
        input_stem = [self.input_stem[0].get_active_subnet(3, False)]
        if self.input_stem_skipping <= 0:
            input_stem.append(ResidualBlock(
                self.input_stem[1].conv.get_active_subnet(self.input_stem[0].active_out_channel, False),
                IdentityLayer(self.input_stem[0].active_out_channel, self.input_stem[0].active_out_channel)
            ))
        input_stem.append(self.input_stem[2].get_active_subnet(self.input_stem[0].active_out_channel, False))
        input_channel = self.input_stem[2].active_out_channel
        latency = latency + loaded_lut[subnet_config['image_size']]['input_stem'][(subnet_config['d'][0], subnet_config['w'][0], subnet_config['w'][0])]

        blocks = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[:len(block_idx) - depth_param]
            for idx in active_idx:
                blocks.append(self.blocks[idx].get_active_subnet(input_channel, False))
                latency = latency + loaded_lut[subnet_config['image_size']]['blocks'][input_channel, self.blocks[idx].active_middle_channels, self.blocks[idx].active_out_channel]
                input_channel = self.blocks[idx].active_out_channel
        classifier = self.classifier.get_active_subnet(input_channel, False)
        latency = latency + loaded_lut[subnet_config['image_size']]['classifier'][input_channel]
        subnet = None
        if verify:
            subnet = ResNets(input_stem, blocks, classifier, max_pooling=self.max_pooling)
            subnet.set_bn_param(**self.get_bn_param())
        return latency, subnet

    def build_single_annette_lut(self, image_size=224):
        """
        Make latency estimations for all sub-elements of the OFA network and store them in a dict.
        This can be used as look-up-table (lut) to get the latency estimations of a ofa-subnet
        Args:
            image_size (): input image size

        Returns: dictionary containing latency estimations for each sub-element in the ofa-network

        """

        # assert (image_size in [128, 160, 192, 224])  # image resolution has to be one of these values
        assert (image_size in [128, 144, 160, 176, 192, 224, 240, 256])  # image resolution has to be one of these values

        annette_latency_predictor = AnnetteLatencyLayerPrediction()

        ##############
        # input stem #
        ##############
        # TODO the solution for the input stem is not nice
        input_stem_latency_dict = {}
        depth_config = [2, 2, 2, 2, 2]
        expand_config = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
                         0.35, 0.35]
        width_config = [2, 2, 2, 2, 2, 2]
        for d0 in [0, 1, 2]:
            for w0 in [0, 1, 2]:
                for w1 in [0, 1, 2]:
                    depth_config[0] = d0
                    width_config[0] = w0
                    width_config[1] = w1
                    self.set_active_subnet(d=depth_config, e=expand_config, w=width_config)
                    input_stem = [self.input_stem[0].get_active_subnet(3, False)]
                    if self.input_stem_skipping <= 0:
                        input_stem.append(ResidualBlock(
                            self.input_stem[1].conv.get_active_subnet(self.input_stem[0].active_out_channel, False),
                            IdentityLayer(self.input_stem[0].active_out_channel, self.input_stem[0].active_out_channel)
                        ))
                    input_stem.append(
                        self.input_stem[2].get_active_subnet(self.input_stem[0].active_out_channel, False))
                    input_stem = InputStemLayers(input_stem)
                    input_stem.set_bn_param(**self.get_bn_param())
                    latency = annette_latency_predictor.predict_efficiency(input_stem, (1, 3, image_size, image_size))

                    # to select input stem d0, w0 and w1 are needed
                    input_stem_latency_dict[(d0, w0, w1)] = latency

        ##########
        # blocks #
        ##########
        blocks_latency_dict = {}

        resolution = int(image_size/4)

        for stage_id, block_idx in enumerate(self.grouped_block_index):
            self.logfile.write('predictions for stage_id: ' + str(stage_id) + '\n')
            new_stage = True

            # all possible width configurations
            for w_in in [0, 1, 2]:
                for w_out in [0, 1, 2]:

                    down_sample_block = True
                    for idx in block_idx:
                        # reset resolution for next iteration
                        if down_sample_block and not new_stage and stage_id != 0:
                            resolution = resolution * 2

                        # all possible expand configurations
                        for e in [0.2, 0.25, 0.35]:

                            # set all parameters
                            input_channel = self.blocks[idx].in_channel_list[w_in]
                            self.blocks[idx].active_out_channel = self.blocks[idx].out_channel_list[w_out]
                            self.blocks[idx].active_expand_ratio = e  # middle channels are calculated from this

                            # get block
                            block = ResNetBlockLayer(self.blocks[idx].get_active_subnet(input_channel, False))
                            # skip redundant blocks
                            if (input_channel, self.blocks[idx].active_middle_channels, self.blocks[idx].active_out_channel) in blocks_latency_dict:
                                continue
                            # make latency prediction
                            latency = annette_latency_predictor.predict_efficiency(block, (1, input_channel, resolution, resolution))
                            blocks_latency_dict[(input_channel, self.blocks[idx].active_middle_channels, self.blocks[idx].active_out_channel)] = latency

                        # parameters for next block
                        if down_sample_block and stage_id != 0:
                            resolution = int(resolution/2)
                        down_sample_block = False
                        new_stage = False

        ##############
        # classifier #
        ##############
        classifier_latency_dict = {}
        for w in [0, 1, 2]:
            input_channel = self.classifier.in_features_list[w]
            classifier = ResNetClassifier(self.classifier.get_active_subnet(input_channel, False))
            latency = annette_latency_predictor.predict_efficiency(classifier, (1, input_channel, int(image_size/32), int(image_size/32)))
            classifier_latency_dict[input_channel] = latency

        return {'input_stem': input_stem_latency_dict, 'blocks': blocks_latency_dict, 'classifier': classifier_latency_dict}

    def build_annette_lut(self, image_sizes=None, save_path='tmp/resnet50_annette_latency_lut.pkl'):
        if image_sizes is None:
            image_sizes = [128, 144, 160, 176, 192, 224, 240, 256]

        annette_latency_lut = {}
        for image_size in image_sizes:
            annette_latency_lut_one_resolution = self.build_single_annette_lut(image_size=image_size)
            annette_latency_lut[image_size] = annette_latency_lut_one_resolution

        # TODO save does not work some missing write attribute bullshit
        # with open('restnet_lut_res' + str(image_size) + '.pkl', 'wb') as save_file:
        #     pickle.dump(annette_latency_lut, save_path)
        #     save_file.close()
        return annette_latency_lut

    def get_active_net_config(self):
        input_stem_config = [self.input_stem[0].get_active_subnet_config(3)]
        if not self.small_input_stem:
            if self.input_stem_skipping <= 0:
                input_stem_config.append({
                    'name': ResidualBlock.__name__,
                    'conv': self.input_stem[1].conv.get_active_subnet_config(self.input_stem[0].active_out_channel),
                    'shortcut': IdentityLayer(self.input_stem[0].active_out_channel, self.input_stem[0].active_out_channel),
                })
            input_stem_config.append(self.input_stem[2].get_active_subnet_config(self.input_stem[0].active_out_channel))
            input_channel = self.input_stem[2].active_out_channel
        else:
            input_channel = self.input_stem[0].active_out_channel

        blocks_config = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[:len(block_idx) - depth_param]
            for idx in active_idx:
                blocks_config.append(self.blocks[idx].get_active_subnet_config(input_channel))
                input_channel = self.blocks[idx].active_out_channel
        classifier_config = self.classifier.get_active_subnet_config(input_channel)
        return {
            'name': ResNets.__name__,
            'bn': self.get_bn_param(),
            'input_stem': input_stem_config,
            'blocks': blocks_config,
            'classifier': classifier_config,
        }

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks:
            block.re_organize_middle_weights(expand_ratio_stage)