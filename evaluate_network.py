import argparse
import os
import torch
import torchvision
from torchvision import transforms, datasets

import ofa.tutorial.imagenet_eval_helper as eval_helper
from ofa.imagenet_classification.elastic_nn.networks.ofa_resnets import OFAResNets, OFASmallResNets
from ofa.imagenet_classification.elastic_nn.networks.ofa_mbv3 import OFAMobileNetV3
from torchviz import make_dot

parser = argparse.ArgumentParser()
parser.add_argument('--net_path', type=str, default='', help='Path to the trained OFA-Model')
parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'imagenette', 'cifar10'])
parser.add_argument('--data_path', type=str, help='Path to dataset')
parser.add_argument('--net', type=str, default='ResNet50', choices=['ResNet50', 'MobileNetV3', 'ResNet18'])
parser.add_argument('--image_size', type=int)
parser.add_argument('--subnet_kernel', type=int, default=None,
                    help='For ResNet: not available\nFor MobileNet: [3, 5, 7]')
parser.add_argument('--subnet_depth', type=int, default=None, help='For ResNet: [0, 1, 2]\nFor MobileNet: [2, 3, 4]')
parser.add_argument('--subnet_expand', type=str, default=None,
                    help='For ResNet: value between 0 and 1\nFor MobileNet: [3, 4, 6]')
parser.add_argument('--subnet_width', type=float, default=None, help='Value between 0 and 1')
parser.add_argument('--subnet_random', type=bool, default=False, help='Set to True to sample and evaluate a random subnetwork')
args = parser.parse_args()

# configure devices
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.num_workers = 2

# parameters for different data sets
if args.dataset == 'imagenet':
    if not args.image_size:
        args.image_size = 224
    if args.data_path is None:
        args.data_path = './dataset/imagenet/'
    args.mean = [0.485, 0.456, 0.406]
    args.std = [0.229, 0.224, 0.225]
    args.n_classes = 1000
elif args.dataset == 'imagenette':
    if not args.image_size:
        args.image_size = 224
    if args.data_path is None:
        args.data_path = './dataset/imagenette2-320/'
    args.mean = [0.485, 0.456, 0.406]
    args.std = [0.229, 0.224, 0.225]
    args.n_classes = 10
elif args.dataset == 'cifar10':
    if not args.image_size:
        args.image_size = 32
    if args.data_path is None:
        args.data_path = './dataset/'
    args.mean = [0.4914, 0.4822, 0.4465]
    args.std = [0.2023, 0.1994, 0.2010]
    args.n_classes = 10
else:
    raise NotImplementedError

args.width_mult_list = [0.65, 0.8, 1.0]

# select network
if args.net == 'ResNet50':
    net = OFAResNets(
        n_classes=args.n_classes,
        depth_list=[0, 1, 2],
        expand_ratio_list=[0.2, 0.25, 0.35],
        width_mult_list=args.width_mult_list
    )
elif args.net == 'MobileNetV3':
    net = OFAMobileNetV3(
        n_classes=args.n_classes,
        ks_list=[3, 5, 7],
        depth_list=[2, 3, 4],
        expand_ratio_list=[3, 4, 6],
        width_mult=args.width_mult_list
    )
elif args.net == 'ResNet18':
    net = OFASmallResNets(
        n_classes=args.n_classes,
        depth_list=[0, 1, 2],
        width_mult_list=args.width_mult_list
    )
else:
    raise NotImplementedError

if args.subnet_expand is not None:
    if args.net == 'ResNet50' or args.net == 'ResNet18':
        args.subnet_expand = float(args.subnet_expand)
    elif args.net == 'MobileNetV3':
        args.subnet_expand = int(args.subnet_expand)
    else:
        raise NotImplementedError


# Create data loader for selected dataset
def build_val_transform(size):
    return transforms.Compose([
        # transforms.Resize(int(math.ceil(size / 0.875))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=args.mean,
            std=args.std,
        ),
    ])


if args.dataset == 'imagenet' or args.dataset == 'imagenette':
    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=os.path.join(args.data_path, 'val'),
            transform=build_val_transform(args.image_size)
        ),
        batch_size=100,  # test batch size
        shuffle=True,
        num_workers=args.num_workers,  # number of workers for the data loader
        pin_memory=True,
        drop_last=False,
    )
elif args.dataset == 'cifar10':
    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=False, download=True, transform=build_val_transform(args.image_size)
        ),
        batch_size=100,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
else:
    raise NotImplementedError

# load network
init = torch.load(args.net_path)['state_dict']
net.load_state_dict(init)

# draw network and save as pdf
# make_dot(net(torch.rand(1, 3, 224, 224))).render("tests/viz_" + args.net)

# create network config (select network or a sub-network)
if not args.subnet_random:
    if args.subnet_depth is None and args.subnet_expand is None and args.subnet_width is None:
        print('Evaluate max net ...')
        net.set_max_net()
    else:
        print('Eval subnet ...')
        arch_config = net.set_active_subnet(d=args.subnet_depth, e=args.subnet_expand, w=args.width_mult_list.index(args.subnet_width))
        print(arch_config)
else:
    print('Eval subnet ...')
    arch_config = net.sample_active_subnet()
    print(arch_config)

# calculate accuracy on test data
top1_acc = eval_helper.validate(net=net,
                                path=args.data_path,
                                image_size=args.image_size,
                                data_loader=data_loader,
                                device=args.device)

# print accuracy
print('\n***Top1 accuracy is: ' + str(top1_acc) + '%***\n')
