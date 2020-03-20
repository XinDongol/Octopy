import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(opt):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.

    Args:
        opt: configurations

    Returns:
        user_groups (dict): {user idx: list of sample idx} 
    """
    print('****', opt.dataset)
    if opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if opt.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(opt.data_dir, train=True, download=True,
                                        transform=apply_transform)

            test_dataset = datasets.CIFAR10(opt.data_dir, train=False, download=True,
                                        transform=apply_transform)
        else:
            train_dataset = datasets.CIFAR100(opt.data_dir, train=True, download=True,
                                        transform=apply_transform)

            test_dataset = datasets.CIFAR100(opt.data_dir, train=False, download=True,
                                        transform=apply_transform)            

        # sample training data amongst users
        if opt.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, opt.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if opt.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, opt.num_users)

    elif opt.dataset == 'mnist' or opt.dataset == 'fmnist':

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(opt.data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(opt.data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if opt.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, opt.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if opt.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, opt.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, opt.num_users)

    return train_dataset, test_dataset, user_groups