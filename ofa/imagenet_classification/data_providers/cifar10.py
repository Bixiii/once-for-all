# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import warnings
import os
import math
from PIL import Image
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .base_provider import DataProvider
from ofa.utils.my_dataloader import MyRandomResizedCrop, MyDistributedSampler


class Cifar10DataProvider(DataProvider):
    DEFAULT_PATH = './dataset/cifar10'

    def __init__(
        self,
        save_path=None,
        train_batch_size=128,
        test_batch_size=128,
        n_worker=0,
        image_size=32,
        valid_size=None,  # unused but here for compatibility
        num_replicas=None,  # unused but here for compatibility
        rank=None,  # unused but here for compatibility
        resize_scale=1,  # unused data augmentation but here for compatibility
        distort_color=None,  # unused data augmentation but here for compatibility
    ):

        warnings.filterwarnings('ignore')
        self._save_path = save_path

        if not isinstance(image_size, int):
            raise TypeError('Elastic resolution for Cifar10 not supported, image size must be of type int')
        self.image_size = image_size  # int or list of int
        self.distort_color = 'None'
        self.resize_scale = 1

        self.active_img_size = self.image_size

        train_dataset = self.train_dataset(self.build_train_transform())
        self.train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            num_workers=n_worker,
            # pin_memory=True,
        )

        test_dataset = self.test_dataset(self.build_valid_transform())
        self.test = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=n_worker,
            # pin_memory=True,
        )

        self.valid = self.test

    @staticmethod
    def name():
        return 'cifar10'

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
            if not os.path.exists(self._save_path):
                os.makedirs(self._save_path)
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())

    def train_dataset(self, _transforms):
        return datasets.CIFAR10(
            self.save_path, train=True, transform=_transforms, download=True
        )

    def test_dataset(self, _transforms):
        return datasets.CIFAR10(
            self.save_path, train=False, transform=_transforms, download=True
        )

    @property
    def normalize(self):
        return transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )

    def build_train_transform(self, image_size=None, print_log=True):
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def build_valid_transform(self, image_size=None):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def assign_active_img_size(self, new_img_size):
        pass  # elastic resolution does not make sense for cifar10

    def build_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        pass  # used to reset bn statistics in validate all resolutions, no elastic resolution in cifar10
