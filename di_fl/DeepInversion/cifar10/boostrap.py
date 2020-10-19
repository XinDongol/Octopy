import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
import torchvision.transforms as transforms

from runx.logx import logx

import numpy as np
import glob
import argparse
from tqdm import tqdm

import utils
import resnet_cifar


# Training settings
parser = argparse.ArgumentParser(description='CIFAR-10 DI Boostrap')
parser.add_argument('--resume', type=int, default=0,
                    help='resume model')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--subset_pct', type=float, default=0.5, metavar='M',
                    help='the percent of data we use before di')   
# di settings
parser.add_argument('--di_batch_size', type=int, default=256, metavar='N',
                    help='batch size for di (default: 256)')
parser.add_argument('--num_di_batch', type=int, default=256, metavar='N',
                    help='batch size for di (default: 256)')


parser.add_argument('--logdir', type=str, default=None,
                    help='target log directory')
args = parser.parse_args()


logx.initialize(logdir=args.logdir, coolname=True, tensorboard=True,
                hparams=vars(args))

torch.manual_seed(args.seed)

device = torch.device("cuda")

#################################
# prepare dataset and subdataset
#################################
full_trainloader, full_testloader = utils.get_standard_cifar10(
    '/workspace/cifar10', batch_size=args.batch_size, test_batch_size=args.test_batch_size,
    num_workers=12
)

len_full_trainset = len(full_trainloader.dataset)
len_sub_trainset  = int(len_full_trainset*args.subset_pct)
sub_trainset = Subset(full_trainloader.dataset, 
                indices=np.random.permutation(len_full_trainset)[:len_sub_trainset])
sub_trainloader = torch.utils.data.DataLoader(
                    sub_trainset, 
                    batch_size=args.batch_size, shuffle=True, num_workers=12, drop_last=True)


####################################
# model, criterion and optimizer
####################################
net = resnet_cifar.ResNet18(num_classes=10)
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, args.epochs)
best_acc = 0.


####################################
# training
####################################
if args.resume==1:
    checkpoint_file = logx.get_best_checkpoint()
    state_dict, _ = logx.load_model(checkpoint_file)
    net.load_state_dict(state_dict)
    _, test_acc = utils.validate(full_testloader, net, criterion, 0, device, 
        print_freq=1e8, logx=logx, write=False)

for epoch in range(args.epochs):
    utils.train(sub_trainloader, net, criterion, optimizer, epoch, device,
        print_freq=1e8, logx=logx)
    _, test_acc = utils.validate(full_testloader, net, criterion, epoch, device, 
        print_freq=1e8, logx=logx)
    utils.save_model(net, optimizer, test_acc, epoch, logx)



###################################
# di process
###################################
import deepinversion_cifar10
data_type = torch.half if 0 else torch.float
inputs = torch.randn((args.di_batch_size, 3, 32, 32), 
            requires_grad=True, device='cuda', dtype=data_type)
optimizer_di = optim.Adam([inputs], lr=0.05)
di_tensor_dir = utils.mkdir(logx.logdir+'/di_tensor')

## Create hooks for feature statistics catching
loss_r_feature_layers = []
for module in net.modules():
    if isinstance(module, nn.BatchNorm2d):
        loss_r_feature_layers.append(
            deepinversion_cifar10.DeepInversionFeatureHook(module))

for di_idx in tqdm(range(args.num_di_batch)):
    next_batch = len(glob.glob("%s/*.pt" % di_tensor_dir))
    di_tensor = deepinversion_cifar10.get_images(net, loss_r_feature_layers,
                bs=args.di_batch_size, epochs=2000, idx=-1, var_scale=2.5e-5,
                net_student=None, prefix=None, competitive_scale=0.0, 
                train_writer = None, global_iteration=None,
                use_amp=False,
                optimizer = optimizer_di, inputs = inputs, 
                bn_reg_scale = 10.0, random_labels = True, l2_coeff=0.0, 
                name_use=logx.logdir, save_image=False)
    torch.save(di_tensor, '%s/%s.pt'%(di_tensor_dir, next_batch))