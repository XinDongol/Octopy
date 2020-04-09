import copy
import torch
from torchvision import datasets, transforms
from sampling import *


def get_dataset(opt):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.

    Args:
        opt: configurations

    Returns:
        user_groups (dict): {user idx: list of sample idx} 
    """
    # print('****', opt.dataset)
    if opt.dataset == 'cifar10' or opt.dataset == 'cifar100':

        if opt.dataset == 'cifar10':
            transform_train = transforms.Compose([                                   
                transforms.RandomCrop(32, padding=4),                                       
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            train_dataset = datasets.CIFAR10(opt.data_dir, train=True, download=True,
                                        transform=transform_train)
            
            transform_test = transforms.Compose([                                           
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            

            test_dataset = datasets.CIFAR10(opt.data_dir, train=False, download=True,
                                        transform=transform_test)
        else:
            train_dataset = datasets.CIFAR100(opt.data_dir, train=True, download=True,
                                        transform=apply_transform)

            test_dataset = datasets.CIFAR100(opt.data_dir, train=False, download=True,
                                        transform=apply_transform)            

        # sample training data amongst users
        if opt.iid:
            # Sample IID user data from CIFAR
            user_groups = split_iid(train_dataset, opt.num_users)
        else:
            # Sample Non-IID user data from CIFAR
            if opt.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = split_noniid(train_dataset, opt.num_users)

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
            user_groups = split_iid(train_dataset, opt.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if opt.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, opt.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, opt.num_users)
                
    elif opt.dataset == 'test':
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(list(range(opt.num_users**2))))
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(list(range(opt.num_users**2))))
        user_groups = {}
        for k in range(opt.num_users):
            user_groups[k] = np.array([k, k**2])
    else:
        raise NotImplementedError()

    return train_dataset, test_dataset, user_groups


if __name__ == '__main__':
    class Opt():
        def __init__(self):
            self.dataset = 'cifar10'
            self.iid = False
            self.unequal = False
            self.num_users = 20
            self.data_dir = '/n/holyscratch01/kung_lab/xin/cifar_data'
            
    opt = Opt()
    
    train_dataset, test_dataset, user_groups = get_dataset(opt)