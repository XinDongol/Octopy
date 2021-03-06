import argparse
import models
import os


class BaseConfig():
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        model_names = sorted(name for name in models.__dict__
                             if not name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))

        # for system
        self.parser.add_argument(
            '--continue_train', action='store_true', help='continue a training')
        self.parser.add_argument('--expr_dir', type=str,
                                 default='./test_run', help='Name of this experiment. It decides where to store logs and models.')
        self.parser.add_argument('--num_gpu', type=int,
                                 default=2, help='Number of GPUs')
        self.parser.add_argument('--num_local_models_per_gpu', type=int,
                                 default=2, help='Number of local models per GPU')
        self.parser.add_argument('--num_users', type=int,
                                 default=600, help='Total number of users')
        self.parser.add_argument('--num_rounds', type=int,
                                 default=1000, help='Number of rounds training federated model')
        self.parser.add_argument('--user_fraction', type=float,
                                 default=0.1, help='fraction of users')
        self.parser.add_argument('--fed_strategy', type=str,
                                 default='Avg', help='fed_strategy: Avg')
        self.parser.add_argument('--model', metavar='MODEL', default='TestNet',
                                 choices=model_names, help='model architecture: ' +
                                 ' | '.join(model_names))
        # for dataset
        self.dataset_config = self.parser.add_argument_group('dataset_group')
        self.dataset_config.add_argument('--dataset', type=str,
                                         default='mnist', choices=['test','mnist', 'fmnist', 'cifar10', 'cifar100'])
        self.dataset_config.add_argument('--data_dir', type=str,
                                         help='dir of dataset')
        self.dataset_config.add_argument('--iid', action='store_true',
                                         help='use iid datasets')
        self.dataset_config.add_argument('--unequal', action='store_true')

        # for user
        self.user_config = self.parser.add_argument_group('user_group')
        self.user_config.add_argument('--local_epoch', type=int,
                                      default=5, help='Number of epochs training local models')
        self.user_config.add_argument('--optimizer', type=str,
                                      default='SGD', help='optimizer type: one of SGD | Adam ')
        self.user_config.add_argument('--lr', type=float,
                                      default=0.001, help='learning rate')
        self.user_config.add_argument('--weights_decay', type=float, 
                                      default=5e-4, help='weights decay')
        self.user_config.add_argument('--momentum', type=float,
                                      default=0, help='momentum for the optimizer')
        self.user_config.add_argument('--loss_func', type=str,
                                      default='CrossEntropyLoss', help='loss function')
        self.user_config.add_argument('--local_batchsize', type=int,
                                      default='32', help='local_batchsize')

        self.initialized = True

    def parse_args(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        mkdir(self.opt.expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(self.opt.expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
