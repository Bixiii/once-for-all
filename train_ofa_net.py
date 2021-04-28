# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import numpy as np
import os
import random

import horovod.torch as hvd
import torch

from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import (
    DynamicSeparableConv2d,
)
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3, OFAResNets, OFASmallResNets
from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import (
    load_models,
)
from ofa.imagenet_classification.networks import MobileNetV3Large, MobileNetV3, SmallResNets, ResNet50D, ResNet50
from ofa.imagenet_classification.run_manager import DistributedImageNetRunConfig
from ofa.imagenet_classification.run_manager.distributed_run_manager import (
    DistributedRunManager,
)
from ofa.utils import MyRandomResizedCrop, download_url

from settings import deactivate_cuda

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="imagenet", choices=["imagenet", "cifar10"]
)
parser.add_argument(
    '--net', type=str, default='MobileNetV3', choices=['MobileNetV3', 'ResNet18', 'ResNet50']
)
parser.add_argument(
    '--teacher_path',
    type=str,
    default='exp/supernet/checkpoint/model_best.pth.tar'
)

parser.add_argument('--pretrained', type=bool, default=False)

parser.add_argument(
    "--task",
    type=str,
    default="depth",
    choices=[
        "supernet",
        "baseline",
        "kernel",
        "depth",
        "expand",
    ],
)
parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
parser.add_argument("--resume", action="store_true")

args = parser.parse_args()

args.image_size = None
if args.task == "supernet":
    args.target_path = "exp/supernet"
    args.dynamic_batch_size = 1
    if args.dataset == 'imagenet':
        args.image_size = '224'
        args.base_lr = 0.08
    elif args.dataset == 'cifar10':
        args.image_size = '32'
        args.base_lr = 0.08
    args.n_epochs = 450
    args.warmup_epochs = 5
    args.warmup_lr = -1
elif args.task == 'baseline':
    args.target_path = "exp/baseline"
    args.dynamic_batch_size = 1
    if args.dataset == 'imagenet':
        args.image_size = '224'
        args.base_lr = 0.08
    elif args.dataset == 'cifar10':
        args.image_size = '32'
        args.base_lr = 0.08
    args.n_epochs = 450
    args.warmup_epochs = 5
    args.warmup_lr = -1
elif args.task == "kernel":
    args.source_path = 'exp/baseline'
    args.target_path = "exp/baseline_2_kernel"
    args.dynamic_batch_size = 1
    args.n_epochs = 120
    args.base_lr = 3e-2
    args.warmup_epochs = 5
    args.warmup_lr = -1
    args.ks_list = "3,5,7"
    args.expand_list = "6"
    args.depth_list = "4"
elif args.task == "depth":
    args.source_path = "exp/baseline_2_kernel/phase2"
    args.target_path = "exp/kernel_2_kernel_depth/phase%d" % args.phase
    args.dynamic_batch_size = 2
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "6"
        args.depth_list = "3,4"
    else:
        args.n_epochs = 120
        args.base_lr = 7.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "6"
        args.depth_list = "2,3,4"
elif args.task == "expand":
    args.source_path = "exp/kernel_2_kernel_depth/phase2"
    args.target_path = "exp/kernel_depth_2_kernel_depth_width/phase%d" % args.phase
    args.dynamic_batch_size = 4
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "4,6"
        args.depth_list = "2,3,4"
    else:
        args.n_epochs = 120
        args.base_lr = 7.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "3,4,6"
        args.depth_list = "2,3,4"
else:
    raise NotImplementedError

args.manual_seed = 0

args.lr_schedule_type = "cosine"

args.valid_size = 10000

args.opt_type = "sgd"
args.momentum = 0.9
args.no_nesterov = False
args.label_smoothing = 0.1
args.no_decay_keys = "bn#bias"
args.fp16_allreduce = False

args.model_init = "he_fout"
args.validation_frequency = 1
args.print_frequency = 10

args.n_worker = 8

if args.dataset == 'imagenet':
    args.distort_color = "tf"
elif args.dataset == 'cifar10':
    args.distort_color = None
else:
    args.distort_color = None

args.continuous_size = True
args.not_sync_distributed_image_size = False

if args.dataset == 'imagenet':
    if args.image_size is None:
        args.image_size = '128,160,192,224'
    args.resize_scale = 0.08
    args.base_batch_size = 64
    args.weight_decay = 3e-5
elif args.dataset == 'cifar10':
    if args.image_size is None:
        args.image_size = '16,24,32'
    args.resize_scale = 1
    args.base_batch_size = 128
    args.weight_decay = 4e-5

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.base_stage_width = "proxyless"

args.width_mult_list = "1.0"
args.independent_distributed_sampling = False

args.kd_type = "ce"

args.dropout = 0.1
# for different params in supernet task
if args.task == 'supernet' or 'baseline':
    args.dy_conv_scaling_mode = -1
    args.kd_ratio = -1.0  # not using teacher model
    args.teacher_model = None
else:
    args.dy_conv_scaling_mode = 1
    args.kd_ratio = 1.0


if __name__ == "__main__":
    os.makedirs(args.target_path, exist_ok=True)

    # Initialize Horovod
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    if not deactivate_cuda:
        torch.cuda.set_device(hvd.local_rank())

    if args.pretrained:
        args.teacher_path = download_url(
            "https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K7",
            model_dir=".torch/ofa_checkpoints/%d" % hvd.rank(),
        )
    elif not args.teacher_path:
        args.teacher_path = 'exp/supernet/checkpoint/model_best.pth.tar'

    num_gpus = hvd.size()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # image size
    args.image_size = [int(img_size) for img_size in args.image_size.split(",")]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        "momentum": args.momentum,
        "nesterov": not args.no_nesterov,
    }
    args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 2
    run_config = DistributedImageNetRunConfig(
        **args.__dict__, num_replicas=num_gpus, rank=hvd.rank()
    )

    # print run config information
    if hvd.rank() == 0:
        print("Run config:")
        for k, v in run_config.config.items():
            print("\t%s: %s" % (k, v))

    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None
    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    if args.task == "supernet":
        if args.net == 'ResNet18':
            net = SmallResNets(
                n_classes=run_config.data_provider.n_classes,
                bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=args.dropout,
                blocks_per_layer_list=[3, 4, 6, 3],  # ResNet34 as teacher model
                width_mult=1.0,
            )
        elif args.net == 'MobileNetV3':
            net = MobileNetV3Large(  # MobileNetV3Large as teacher model
                n_classes=run_config.data_provider.n_classes,
                bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=args.dropout,
                width_mult=1.0,
                ks=7,
                expand_ratio=6,
                depth_param=4,
            )
        elif args.net == 'ResNet50':
            net = ResNet50D(  # ResNet50D (dense) as teacher model
                n_classes=run_config.data_provider.n_classes,
                bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=args.dropout,
                width_mult=1.0,
            )
    elif args.task == 'baseline':
        if args.net == 'ResNet18':
            net = SmallResNets(
                n_classes=run_config.data_provider.n_classes,
                bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=args.dropout,
                blocks_per_layer_list=[2, 2, 2, 2],
                width_mult=1.0,
            )
        elif args.net == 'MobileNetV3':
            raise NotImplementedError
        elif args.net == 'ResNet50':
            net = ResNet50(
                n_classes=run_config.data_provider.n_classes,
                bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=args.dropout,
                width_mult=1.0,
            )

    else:
        # build net from args
        args.width_mult_list = [
            float(width_mult) for width_mult in args.width_mult_list.split(",")
        ]
        args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
        args.expand_list = [int(e) for e in args.expand_list.split(",")]
        args.depth_list = [int(d) for d in args.depth_list.split(",")]

        args.width_mult_list = (
            args.width_mult_list[0]
            if len(args.width_mult_list) == 1
            else args.width_mult_list
        )
        if args.net == 'ResNet18':
            net = OFASmallResNets(
                n_classes=run_config.data_provider.n_classes,
                bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=args.dropout,
                depth_list=args.depth_list,
                width_mult_list=args.width_mult_list
            )
            pass
        elif args.net == 'MobileNetV3':
            net = OFAMobileNetV3(
                n_classes=run_config.data_provider.n_classes,
                bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=args.dropout,
                base_stage_width=args.base_stage_width,
                width_mult=args.width_mult_list,
                ks_list=args.ks_list,
                expand_ratio_list=args.expand_list,
                depth_list=args.depth_list,
            )
        elif args.net == 'ResNet50':
            net = OFAResNets(
                n_classes=run_config.data_provider.n_classes,
                bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=args.dropout,
                expand_ratio_list=args.expand_list,
                depth_list=args.depth_list,
                width_mult_list=args.width_mult_list
            )

    # teacher model
    if args.kd_ratio > 0:
        if args.net == 'ResNet18':
            args.teacher_model = SmallResNets(
                n_classes=run_config.data_provider.n_classes,
                bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=args.dropout,
                blocks_per_layer_list=[3, 4, 6, 3],  # ResNet34 as teacher model
                width_mult=1.0,
            )
        elif args.net == 'MobileNetV3':
            args.teacher_model = MobileNetV3Large(
                n_classes=run_config.data_provider.n_classes,
                bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=0,
                width_mult=1.0,
                ks=7,
                expand_ratio=6,
                depth_param=4,
            )
        elif args.net == 'ResNet50':
            args.teacher_model = ResNet50D(  # ResNet50D (dense) as teacher model
                n_classes=run_config.data_provider.n_classes,
                bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=args.dropout,
                width_mult=1.0,
            )

        args.teacher_model.cuda()

    """ Distributed RunManager """
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    distributed_run_manager = DistributedRunManager(
        args.target_path,
        net,
        run_config,
        compression,
        backward_steps=args.dynamic_batch_size,
        is_root=(hvd.rank() == 0),
    )
    distributed_run_manager.save_config()
    # hvd broadcast
    distributed_run_manager.broadcast()

    # load teacher net weights
    if args.kd_ratio > 0:
        load_models(
            distributed_run_manager, args.teacher_model, model_path=args.teacher_path
        )

    # training supernet
    if args.task == "supernet" or 'baseline':
        distributed_run_manager.train(
            args, warmup_epochs=args.warmup_epochs, warmup_lr=args.warmup_lr
        )

    # training progressive shrinking
    else:
        from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import (
            train,
            validate,
        )

        validate_func_dict = {
            "image_size_list": {224} if args.dataset == 'imagenet' else {32}
            if isinstance(args.image_size, int)
            else sorted({160, 224}),
            "ks_list": sorted({min(args.ks_list), max(args.ks_list)}),
            "expand_ratio_list": sorted({min(args.expand_list), max(args.expand_list)}),
            "depth_list": sorted({min(net.depth_list), max(net.depth_list)}),
        }
        if args.task == "kernel":
            validate_func_dict["ks_list"] = sorted(args.ks_list)
            if distributed_run_manager.start_epoch == 0:
                if args.pretrained:
                    args.ofa_checkpoint_path = download_url(
                        "https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K7",
                        model_dir=".torch/ofa_checkpoints/%d" % hvd.rank(),
                    )
                else:
                    args.ofa_checkpoint_path = os.path.join(args.source_path, 'checkpoint/model_best.pth.tar')

                load_models(
                    distributed_run_manager,
                    distributed_run_manager.net,
                    args.ofa_checkpoint_path,
                )
                distributed_run_manager.write_log(
                    "%.3f\t%.3f\t%.3f\t%s"
                    % validate(
                        distributed_run_manager, is_test=True, **validate_func_dict
                    ),
                    "valid",
                )
            else:
                assert args.resume
            train(
                distributed_run_manager,
                args,
                lambda _run_manager, epoch, is_test: validate(
                    _run_manager, epoch, is_test, **validate_func_dict
                ),
            )
        elif args.task == "depth":
            from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import (
                train_elastic_depth,
            )  # noqa

            if args.phase == 1:
                if args.pretrained:
                    args.ofa_checkpoint_path = download_url(
                        "https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K357",  # noqa
                        model_dir=".torch/ofa_checkpoints/%d" % hvd.rank(),
                    )
                else:
                    args.ofa_checkpoint_path = os.path.join(args.source_path, 'checkpoint/model_best.pth.tar')
            else:
                if args.pretrained:
                    args.ofa_checkpoint_path = download_url(
                        "https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D34_E6_K357",  # noqa
                        model_dir=".torch/ofa_checkpoints/%d" % hvd.rank(),
                    )
                else:
                    args.ofa_checkpoint_path = os.path.join(args.source_path, 'checkpoint/model_best.pth.tar')

            train_elastic_depth(
                train, distributed_run_manager, args, validate_func_dict
            )
        elif args.task == "expand":
            from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import (
                train_elastic_expand,
            )  # noqa

            if args.phase == 1:
                if args.pretrained:
                    args.ofa_checkpoint_path = download_url(
                        "https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D234_E6_K357",  # noqa
                        model_dir=".torch/ofa_checkpoints/%d" % hvd.rank(),
                    )
                else:
                    args.ofa_checkpoint_path = os.path.join(args.source_path, 'checkpoint/model_best.pth.tar')
            else:
                if args.pretrained:
                    args.ofa_checkpoint_path = download_url(
                        "https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D234_E46_K357",  # noqa
                        model_dir=".torch/ofa_checkpoints/%d" % hvd.rank(),
                    )
                else:
                    args.ofa_checkpoint_path = os.path.join(args.source_path, 'checkpoint/model_best.pth.tar')
            train_elastic_expand(
                train, distributed_run_manager, args, validate_func_dict
            )
        else:
            raise NotImplementedError
