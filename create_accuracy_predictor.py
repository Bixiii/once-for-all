import argparse
import glob
import os

import torch
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager, DistributedRunManager
from ofa.imagenet_classification.elastic_nn.networks import OFAResNets
from ofa.nas.accuracy_predictor import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, default='imagenet',
    choices=[
        'imagenet', 'cifar10'
    ]
)
parser.add_argument(
    '--data_path', type=str, default=None, help='Path to dataset'
)
parser.add_argument(
    '--net', type=str, default='ResNet50',
    choices=[
        'ResNet50',
    ]
)
parser.add_argument('--net_path', default='', help='Path where network is stored')
parser.add_argument(
    '--experiment_id', type=str, default='', help='Id to identify the experiment'
)
args = parser.parse_args()

if not args.net_path:
    args.net_path = 'exp/exp_OFA' + args.net + '_' + args.dataset + '_' + args.experiment_id \
                    + '/kernel_depth_expand-2-kernel_depth_expand_width/phase2/checkpoint/model_best.pth.tar'
    print('No network path given, try to load network form default location: ', args.net_path)

# dataset parameters
if args.dataset == 'cifar10':
    args.image_size = 32
    args.image_size_list = [32]
    args.num_classes = 10
    args.acc_dataset_size = 4000
    args.small_input_stem = True
    # Predictor architecture
    num_layers = 1
    num_hidden_units = 100
else:
    args.image_size = 224
    args.image_size_list = [128, 160, 192, 224]
    args.num_classes = 1000
    args.acc_dataset_size = 16000
    args.small_input_stem = False
    # Predictor architecture
    num_layers = 3
    num_hidden_units = 400

# training parameters
batch_size = 100
n_workers = 0
args.lr = 1e-3
args.weight_decay = 1e-4
args.num_epochs = 200
if torch.cuda.is_available():
    device = 'cuda'
else:
    raise EnvironmentError('GPU needed')

# OFA parameters
depth_list = [0, 1, 2]
expand_ratio_list = [0.2, 0.25, 0.35]
width_mult_list = [0.65, 0.8, 1.0]


args.acc_dataset_folder = 'exp/exp_OFA' + args.net + '_' + args.dataset + '_' + args.experiment_id + '/acc_dataset'
comment = '_pt_OFA' + args.net + 'AccuracyPredictor_' + args.dataset + '_' + str(args.experiment_id)

ofa_network = OFAResNets(
    n_classes=args.num_classes,
    dropout_rate=0,
    depth_list=depth_list,
    expand_ratio_list=expand_ratio_list,
    width_mult_list=width_mult_list,
    small_input_stem=args.small_input_stem,
    dataset=args.dataset
)

init = torch.load(args.net_path, map_location='cpu')['state_dict']
ofa_network.load_state_dict(init)
ofa_network.to(device)

run_config = ImagenetRunConfig(test_batch_size=batch_size, n_worker=n_workers, dataset=args.dataset,
                               data_path=args.data_path)
run_manager = RunManager('.tmp/eval_subnet', ofa_network, run_config, init=False)

arch_encoder = ResNetArchEncoder(
    image_size_list=args.image_size_list,
    depth_list=depth_list,
    expand_list=expand_ratio_list,
    width_mult_list=width_mult_list,
    small_input_stem=True,
)

accuracy_dataset = AccuracyDataset(args.acc_dataset_folder)

# skip creatoin of accuracy dataset if there is already one in the folder
files = glob.glob(os.path.join(args.acc_dataset_folder, 'src/*'))
if len(files) < 1:
    # create accuracy dataset for each image size
    accuracy_dataset.build_acc_dataset(run_manager, ofa_network, args.acc_dataset_size, args.image_size_list)
if not os.path.isfile(os.path.join(args.acc_dataset_folder, 'acc.dict')):
    # merge accuracy dataset together
    accuracy_dataset.merge_acc_dataset(args.image_size_list)

# get data loaders for the accuracy dataset
train_data_loader, valid_data_loader, base_acc = accuracy_dataset.build_acc_data_loader(
    arch_encoder=arch_encoder, batch_size=batch_size, n_workers=n_workers
)

accuracy_predictor = AccuracyPredictor(
    arch_encoder=arch_encoder,
    hidden_size=num_hidden_units,
    n_layers=num_layers,
    base_acc=base_acc
)
accuracy_predictor = torch.nn.DataParallel(accuracy_predictor)

train_criterion = torch.nn.MSELoss()
test_criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(accuracy_predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
tensorboard_writer = SummaryWriter(comment=comment)


def train_one_epoch(epoch):
    accuracy_predictor.train()
    train_loss = 0
    with tqdm(total=len(train_data_loader),
              desc='{} Train Epoch #{}'.format(args.dataset, epoch + 1)) as t:
        for i, (inputs, targets) in enumerate(train_data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            output = accuracy_predictor(inputs)
            loss = train_criterion(output, targets)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            train_loss += loss.item()

            t.set_postfix({
                'loss': train_loss,
            })
            t.update(1)
    return train_loss


def test(epoch):
    accuracy_predictor.eval()
    test_loss = 0
    with torch.no_grad():
        with tqdm(total=len(valid_data_loader),
                  desc='Validate Epoch #{}'.format(epoch + 1)) as t:
            for i, (inputs, targets) in enumerate(valid_data_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                # compute output
                output = accuracy_predictor(inputs)
                test_loss += test_criterion(output, targets).item()

                t.set_postfix({
                    'test_loss': test_loss
                })
                t.update(1)
    return test_loss


def save_model():
    checkpoint = {'state_dict': accuracy_predictor.module.state_dict(), 'dataset': args.dataset}
    best_path = os.path.join(args.acc_dataset_folder, 'acc_predictor_model_best.pth.tar')
    torch.save({'state_dict': checkpoint['state_dict']}, best_path)


def train_accuracy_predictor():
    best_test_loss = float('inf')
    for epoch in range(0, args.num_epochs):
        train_loss = train_one_epoch(epoch)
        test_loss = test(epoch)
        scheduler.step()
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_model()
        tensorboard_writer.add_scalar('test loss', test_loss, epoch)
        tensorboard_writer.add_scalar('train loss', train_loss, epoch)


train_accuracy_predictor()
