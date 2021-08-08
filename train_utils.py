from ofa.utils import download_url


def set_progressive_shringking_paramters(args):
    """
    Set the parameters for training and progressive shrinking dependent on the chosen network, task and phase

    The parameters are added to the args object

    """


    info = ''
    if args.use_hvd:
        import horovod.torch as hvd

    args.image_size = None
    args.width_mult_list = None

    if args.task == 'basenet':
        info += 'Set parameter for training basenet\n'
        args.source_path = 'None'
        args.target_path = args.experiment_folder + 'normal'
        args.dynamic_batch_size = 1
        args.n_epochs = 450
        if args.dataset == 'imagenet':
            args.base_lr = 4e-2
        elif args.dataset == 'cifar10':
            args.base_lr = 8e-2
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.phase = 0
        if args.net == 'MobileNetV3':
            args.ks_list = '7'
            args.expand_list = '6'
            args.depth_list = '4'
        elif args.net.__contains__('ResNet'):
            args.ks_list = '3'
            args.width_mult_list = '1.0'
            args.expand_list = '0.35'
            args.depth_list = '2'

    elif args.task == 'kernel':
        info += 'Set parameter for training elastic kernel\n'
        if args.pretrained:
            if args.use_hvd:
                args.source_path = download_url(
                    'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K7',
                    model_dir='.torch/ofa_checkpoints/%d' % hvd.rank()
                )
            else:
                args.source_path = download_url(
                    'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K7',
                    model_dir='.torch/ofa_checkpoints/'
                )
        else:
            args.source_path = args.experiment_folder + 'normal/checkpoint/model_best.pth.tar'
        args.target_path = args.experiment_folder + 'normal-2-kernel'
        args.dynamic_batch_size = 1
        args.n_epochs = 120
        args.base_lr = 3e-2
        args.warmup_epochs = 5
        args.warmup_lr = -1
        if args.net == 'MobileNetV3':
            args.ks_list = '3,5,7'
            args.expand_list = '6'
            args.depth_list = '4'
        if args.net.__contains__('ResNet'):
            raise NotImplementedError('Elastic kernel is not supported for ResNets')

    elif args.task == 'depth':
        info += 'Set parameter for training elastic depth\n'
        if args.net == 'MobileNetV3':
            args.target_path = args.experiment_folder + 'kernel-2-kernel_depth/phase%d' % args.phase
        elif args.net.__contains__('ResNet'):
            args.target_path = args.experiment_folder + 'normal-2-depth/phase%d' % args.phase
        args.dynamic_batch_size = 2
        if args.phase == 1:
            if args.net == 'MobileNetV3':
                if args.pretrained:
                    if args.use_hvd:
                        args.source_path = download_url(
                            'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K357',
                            model_dir='.torch/ofa_checkpoints/%d' % hvd.rank()
                        )
                    else:
                        args.source_path = download_url(
                            'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K357',
                            model_dir='.torch/ofa_checkpoints/'
                        )
                else:
                    args.source_path = args.experiment_folder + 'normal-2-kernel/checkpoint/model_best.pth.tar'
            elif args.net.__contains__('ResNet'):
                args.source_path = args.experiment_folder + 'normal/checkpoint/model_best.pth.tar'
            args.n_epochs = 25
            args.base_lr = 2.5e-3
            args.warmup_epochs = 0
            args.warmup_lr = -1
            if args.net == 'MobileNetV3':
                args.ks_list = '3,5,7'
                args.expand_list = '6'
                args.depth_list = '3,4'
            elif args.net.__contains__('ResNet'):
                args.ks_list = '3'
                args.width_mult_list = '1.0'
                args.expand_list = '0.35'
                args.depth_list = '1,2'
        else:
            if args.net == 'MobileNetV3':
                if args.pretrained:
                    if args.use_hvd:
                        args.source_path = download_url(
                            'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D34_E6_K357',
                            model_dir='.torch/ofa_checkpoints/%d' % hvd.rank()
                        )
                    else:
                        args.source_path = download_url(
                            'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D34_E6_K357',
                            model_dir='.torch/ofa_checkpoints/'
                        )
                else:
                    args.source_path = args.experiment_folder + 'kernel-2-kernel_depth/phase1/checkpoint/model_best.pth.tar'
            elif args.net.__contains__('ResNet'):
                args.source_path = args.experiment_folder + 'normal-2-depth/phase1/checkpoint/model_best.pth.tar'

            args.n_epochs = 120
            args.base_lr = 7.5e-3
            args.warmup_epochs = 5
            args.warmup_lr = -1
            if args.net == 'MobileNetV3':
                args.ks_list = '3,5,7'
                args.expand_list = '6'
                args.depth_list = '2,3,4'
            elif args.net.__contains__('ResNet'):
                args.ks_list = '3'
                args.width_mult_list = '1.0'
                args.expand_list = '0.35'
                args.depth_list = '0,1,2'

    elif args.task == 'expand':
        info += 'Set parameter for training elastic expand\n'
        if args.net == 'MobileNetV3':
            args.target_path = args.experiment_folder + 'kernel_depth-2-kernel_depth_expand/phase%d' % args.phase
        elif args.net == 'ResNet50':
            args.target_path = args.experiment_folder + 'depth-2-depth_expand/phase%d' % args.phase
        elif args.net == 'ResNet34':
            raise NotImplementedError('Elastic expand not supported for ResNet34')

        args.dynamic_batch_size = 4
        if args.phase == 1:
            if args.net == 'MobileNetV3':
                if args.pretrained:
                    if args.use_hvd:
                        args.source_path = download_url(
                            'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D234_E6_K357',
                            model_dir='.torch/ofa_checkpoints/%d' % hvd.rank()
                        )
                    else:
                        args.source_path = download_url(
                            'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D234_E6_K357',
                            model_dir='.torch/ofa_checkpoints/'
                        )
                else:
                    args.source_path = args.experiment_folder + 'kernel-2-kernel_depth/phase2/checkpoint/model_best.pth.tar'
            elif args.net == 'ResNet50':
                args.source_path = args.experiment_folder + 'normal-2-depth/phase2/checkpoint/model_best.pth.tar'

            args.n_epochs = 25
            args.base_lr = 2.5e-3
            args.warmup_epochs = 0
            args.warmup_lr = -1
            if args.net == 'MobileNetV3':
                args.ks_list = '3,5,7'
                args.expand_list = '4,6'
                args.depth_list = '2,3,4'
            elif args.net == 'ResNet50':
                args.ks_list = '3'
                args.width_mult_list = '1.0'
                args.expand_list = '0.25,0.35'
                args.depth_list = '0,1,2'
            elif args.net == 'ResNet34':
                raise NotImplementedError('Elastic expand not supported for ResNet34')
        else:
            if args.net == 'MobileNetV3':
                if args.pretrained:
                    if args.use_hvd:
                        args.source_path = download_url(
                            'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D234_E46_K357',
                            model_dir='.torch/ofa_checkpoints/%d' % hvd.rank()
                        )
                    else:
                        args.source_path = download_url(
                            'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D234_E46_K357',
                            model_dir='.torch/ofa_checkpoints/'
                        )
                else:
                    args.source_path = args.experiment_folder + 'kernel_depth-2-kernel_depth_expand/phase1/checkpoint/model_best.pth.tar'
            elif args.net == 'ResNet50':
                args.source_path = args.experiment_folder + 'depth-2-depth_expand/phase1/checkpoint/model_best.pth.tar'

            args.n_epochs = 120
            args.base_lr = 7.5e-3
            args.warmup_epochs = 5
            args.warmup_lr = -1
            if args.net == 'MobileNetV3':
                args.ks_list = '3,5,7'
                args.expand_list = '3,4,6'
                args.depth_list = '2,3,4'
            elif args.net == 'ResNet50':
                args.ks_list = '3'
                args.width_mult_list = '1.0'
                args.expand_list = '0.20,0.25,0.35'
                args.depth_list = '0,1,2'
            elif args.net == 'ResNet34':
                raise NotImplementedError('Elastic expand not supported for ResNet34')

    elif args.task == 'width':
        info += 'Set parameter for training elastic width\n'
        if args.net == 'MobileNetV3':
            args.target_path = args.experiment_folder + 'kernel_depth_expand-2-kernel_depth_expand_width/phase%d' % args.phase
        elif args.net == 'ResNet50':
            args.target_path = args.experiment_folder + 'depth_expand-2-depth_expand_width/phase%d' % args.phase
        elif args.net == 'ResNet34':
            args.target_path = args.experiment_folder + 'depth-2-depth_width/phase%d' % args.phase

        args.dynamic_batch_size = 4
        if args.phase == 1:
            if args.pretrained:
                raise NotImplementedError
            else:
                if args.net == 'MobileNetV3':
                    args.source_path = args.experiment_folder + 'kernel_depth-2-kernel_depth_expand/phase2/checkpoint/model_best.pth.tar'
                elif args.net == 'ResNet50':
                    args.source_path = args.experiment_folder + 'depth-2-depth_expand/phase2/checkpoint/model_best.pth.tar'
                elif args.net == 'ResNet34':
                    args.source_path = args.experiment_folder + 'normal-2-depth/phase2/checkpoint/model_best.pth.tar'
            args.n_epochs = 25
            args.base_lr = 2.5e-3
            args.warmup_epochs = 0
            args.warmup_lr = -1
            if args.net == 'MobileNetV3':
                raise NotImplementedError('Elastic width multiplier not supported for MobileNetV3')
            elif args.net.__contains__('ResNet'):
                args.ks_list = '3'
                args.width_mult_list = '0.8,1.0'
                args.expand_list = '0.20,0.25,0.35'
                args.depth_list = '0,1,2'
        else:
            if args.pretrained:
                raise NotImplementedError
            else:
                if args.net == 'ResNet50':
                    args.source_path = args.experiment_folder + 'depth_expand-2-depth_expand_width/phase1/checkpoint/model_best.pth.tar'
                elif args.net == 'ResNet34':
                    args.source_path = args.experiment_folder + 'depth-2-depth_width/phase1/checkpoint/model_best.pth.tar'
            args.n_epochs = 120
            args.base_lr = 7.5e-3
            args.warmup_epochs = 5
            args.warmup_lr = -1
            if args.net == 'MobileNetV3':
                raise NotImplementedError('Elastic width multiplier not supported for MobileNetV3')
                # args.ks_list = '3,5,7'
                # args.expand_list = '3,4,6'
                # args.depth_list = '2,3,4'
            elif args.net.__contains__('ResNet'):
                args.ks_list = '3'
                args.width_mult_list = '0.65,0.8,1.0'
                args.expand_list = '0.20,0.25,0.35'
                args.depth_list = '0,1,2'
    else:
        raise NotImplementedError

    info += 'source path: ' + args.source_path + '\n'
    info += 'target path: ' + args.target_path + '\n'
    args.log_info = info
    return args
