import random

from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import DynamicConvLayer, DynamicLinearLayer
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import DynamicResNetBottleneckBlock, DynamicResNetBasicBlock
from ofa.utils.layers import IdentityLayer, ResidualBlock
from ofa.imagenet_classification.networks import ResNets
from ofa.utils import make_divisible, val2list, MyNetwork

__all__ = ['OFAResNets', 'OFAResNet50', 'OFAResNet34']


class OFAResNets(ResNets):

    def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0,
                 depth_list=2, expand_ratio_list=0.25, width_mult_list=1.0,
                 dataset='imagenet', small_input_stem=None, downsample_mode='', block_typ=None, base_depth_list=None,
                 base_width_list=None, max_pooling=None):

        self.depth_list = val2list(depth_list)
        self.expand_ratio_list = val2list(expand_ratio_list)
        self.width_mult_list = val2list(width_mult_list)
        # sort
        self.depth_list.sort()
        self.expand_ratio_list.sort()
        self.width_mult_list.sort()

        # net parameter
        self.dataset = dataset
        self.block_typ = DynamicResNetBottleneckBlock if not block_typ else block_typ
        self.base_depth_list = ResNets.BASE_DEPTH_LIST.copy() if not base_depth_list else base_depth_list
        self.stage_width_list = ResNets.STAGE_WIDTH_LIST.copy() if not base_width_list else base_width_list

        input_channel = [
            make_divisible(64 * width_mult, MyNetwork.CHANNEL_DIVISIBLE) for width_mult in self.width_mult_list
        ]
        mid_input_channel = [
            make_divisible(channel // 2, MyNetwork.CHANNEL_DIVISIBLE) for channel in input_channel
        ]

        for i, width in enumerate(self.stage_width_list):
            self.stage_width_list[i] = [
                make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE) for width_mult in self.width_mult_list
            ]

        n_block_list = [base_depth + max(self.depth_list) for base_depth in base_depth_list]
        stride_list = [1, 2, 2, 2]

        if dataset == 'imagenet':
            input_stem_kernel_size = 7
            input_stem_stride = 2
            self.max_pooling = True if not max_pooling else max_pooling
            self.downsample_mode = 'avgpool_conv' if not downsample_mode else downsample_mode
            self.small_input_stem = False if not small_input_stem else small_input_stem
        elif dataset == 'cifar10':
            input_stem_kernel_size = 3
            input_stem_stride = 1
            self.max_pooling = False if not max_pooling else max_pooling
            self.downsample_mode = 'conv' if not downsample_mode else downsample_mode
            self.small_input_stem = True if not small_input_stem else small_input_stem

        # build input stem
        if self.small_input_stem:
            input_stem = [
                DynamicConvLayer(val2list(3), input_channel, input_stem_kernel_size, stride=input_stem_stride,
                                 use_bn=True, act_func='relu'),
            ]
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
        for d, width, s in zip(n_block_list, self.stage_width_list, stride_list):
            for i in range(d):
                stride = s if i == 0 else 1
                bottleneck_block = block_typ(
                    input_channel, width, expand_ratio_list=self.expand_ratio_list,
                    kernel_size=3, stride=stride, act_func='relu', downsample_mode=self.downsample_mode,
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
        if not self.small_input_stem:
            if self.input_stem_skipping <= 0:
                input_stem.append(ResidualBlock(
                    self.input_stem[1].conv.get_active_subnet(self.input_stem[0].active_out_channel, preserve_weight),
                    IdentityLayer(self.input_stem[0].active_out_channel, self.input_stem[0].active_out_channel)
                ))
            input_stem.append(
                self.input_stem[2].get_active_subnet(self.input_stem[0].active_out_channel, preserve_weight))
            input_channel = self.input_stem[2].active_out_channel
        else:
            input_channel = self.input_stem[0].active_out_channel

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

    def get_active_net_config(self):
        input_stem_config = [self.input_stem[0].get_active_subnet_config(3)]
        if not self.small_input_stem:
            if self.input_stem_skipping <= 0:
                input_stem_config.append({
                    'name': ResidualBlock.__name__,
                    'conv': self.input_stem[1].conv.get_active_subnet_config(self.input_stem[0].active_out_channel),
                    'shortcut': IdentityLayer(self.input_stem[0].active_out_channel,
                                              self.input_stem[0].active_out_channel),
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


class OFAResNet50(OFAResNets):

    def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0,
                 depth_list=2, expand_ratio_list=0.25, width_mult_list=1.0,
                 dataset='imagenet', small_input_stem=None, downsample_mode=''):

        self.block_typ = DynamicResNetBottleneckBlock
        self.base_depth_list = [2, 2, 4, 2]

        super(OFAResNet50, self).__init__(n_classes, bn_param, dropout_rate, depth_list, expand_ratio_list,
                                          width_mult_list, dataset, small_input_stem, downsample_mode, self.block_typ,
                                          self.base_depth_list)


class OFAResNet34(OFAResNets):

    def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0,
                 depth_list=2, expand_ratio_list=0.25, width_mult_list=1.0,
                 dataset='imagenet', small_input_stem=None, downsample_mode=''):

        self.block_typ = DynamicResNetBasicBlock
        self.base_depth_list = [1, 2, 4, 2]
        self.base_width_list = [64, 128, 256, 512]
        self.downsample_mode = 'conv' if not downsample_mode else downsample_mode

        super(OFAResNet34, self).__init__(n_classes, bn_param=bn_param, dropout_rate=dropout_rate,
                                          depth_list=depth_list, expand_ratio_list=expand_ratio_list,
                                          width_mult_list=width_mult_list, dataset=dataset,
                                          small_input_stem=small_input_stem, downsample_mode=self.downsample_mode,
                                          block_typ=self.block_typ, base_depth_list=self.base_depth_list,
                                          base_width_list=self.base_width_list)
