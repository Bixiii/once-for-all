# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from ofa.utils import calc_learning_rate, build_optimizer
from ofa.imagenet_classification.data_providers import *

__all__ = ['MyRunConfig', 'MyImagenetRunConfig']


class MyRunConfig:
    def __init__(
        self,
        n_epochs,
        init_lr,
        lr_schedule_type,
        lr_schedule_param,
        dataset,
        data_path,
        train_batch_size,
        test_batch_size,
        valid_size,
        opt_type,
        opt_param,
        momentum,
        weight_decay,
        label_smoothing,
        no_decay_keys,
        mixup_alpha,
        model_init,
        validation_frequency,
        print_frequency,
    ):
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        self.mixup_alpha = mixup_alpha

        self.model_init = model_init
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return MyRunConfig(**self.config)

    """ data provider """

    @property
    def data_provider(self):
        raise NotImplementedError

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test

    def random_sub_train_loader(
        self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None
    ):
        return self.data_provider.build_sub_train_loader(
            n_images, batch_size, num_worker, num_replicas, rank
        )


class MyImagenetRunConfig(MyRunConfig):
    def __init__(
        self,
        n_epochs=150,
        init_lr=0.05,
        lr_schedule_type='cosine',
        lr_schedule_param=None,
        dataset='imagenet',
        data_path=None,
        train_batch_size=256,
        test_batch_size=500,
        valid_size=None,
        opt_type='sgd',
        opt_param=None,
        momentum=0.9,
        weight_decay=4e-5,
        label_smoothing=0.1,
        no_decay_keys=None,
        mixup_alpha=None,
        model_init='he_fout',
        validation_frequency=1,
        print_frequency=10,
        n_worker=32,
        resize_scale=0.08,
        distort_color='tf',
        image_size=224,
        **kwargs
    ):
        super(MyImagenetRunConfig, self).__init__(
            n_epochs,
            init_lr,
            lr_schedule_type,
            lr_schedule_param,
            dataset,
            data_path,
            train_batch_size,
            test_batch_size,
            valid_size,
            opt_type,
            opt_param,
            momentum,
            weight_decay,
            label_smoothing,
            no_decay_keys,
            mixup_alpha,
            model_init,
            validation_frequency,
            print_frequency,
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == ImagenetDataProvider.name():
                DataProviderClass = ImagenetDataProvider
            elif self.dataset == Cifar10DataProvider.name():
                DataProviderClass = Cifar10DataProvider
            elif self.dataset == ImagenetteDataProvider.name():
                DataProviderClass = ImagenetteDataProvider
            else:
                raise NotImplementedError
            self.__dict__["_data_provider"] = DataProviderClass(
                save_path=self.data_path,
                train_batch_size=self.train_batch_size,
                test_batch_size=self.test_batch_size,
                valid_size=self.valid_size,
                n_worker=self.n_worker,
                resize_scale=self.resize_scale,
                distort_color=self.distort_color,
                image_size=self.image_size,
            )
        return self.__dict__["_data_provider"]