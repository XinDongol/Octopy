import time
import copy
from collections import OrderedDict
from argparse import ArgumentParser
import json
import random 

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import fl_data
import utils
import resnet_cifar

from runx.logx import logx

parser = ArgumentParser()
parser.add_argument('--logdir', type=str)
parser.add_argument('--rounds', type=int)
parser.add_argument('--num_devices', type=int)
parser.add_argument('--device_pct', type=float)
parser.add_argument('--non_iid', type=int, choices=[0, 1])
# local setting
parser.add_argument('--local_epochs', type=int)
parser.add_argument('--local_lr', type=float)
parser.add_argument('--local_bsz', type=int)

# di process
parser.add_argument('--di_bn', type=int, choices=[0,1])
parser.add_argument('--di_weights', type=int, choices=[0,1])
parser.add_argument('--iterations_per_layer', type=int)
parser.add_argument('--de_batch_size', type=int)

args = parser.parse_args()

logx.initialize(logdir=args.logdir, coolname=True, tensorboard=True,
                hparams=vars(args))

writer = SummaryWriter(args.logdir)


##########################
# trainset and testset
##########################

transform_train = transforms.Compose([                                   
    transforms.RandomCrop(32, padding=4),                                       
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='/cifar', train=True, 
                                        download=True,
                                        transform=transform_train)

transform_test = transforms.Compose([                                           
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='/cifar', train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False,
                                         num_workers=2)



def create_device(net, device_id, trainset, data_idxs, lr=0.1,
                  milestones=None, batch_size=128):
    if milestones == None:
        milestones = [25, 50, 65]

    device_net = copy.deepcopy(net)
    optimizer = torch.optim.SGD(device_net.parameters(), lr=lr, momentum=0.9,
                                weight_decay=5e-4)
    # optimizer = torch.optim.Adam(device_net.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=milestones,
                                                     gamma=0.1)
    device_trainset = fl_data.DatasetSplit(trainset, data_idxs)
    device_trainloader = torch.utils.data.DataLoader(device_trainset,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     num_workers=8)
    return {
        'net': device_net,
        'id': device_id,
        'dataloader': device_trainloader, 
        'optimizer': optimizer,
        'scheduler': scheduler,
        'train_loss_tracker': [],
        'train_acc_tracker': [],
        'test_loss_tracker': [],
        'test_acc_tracker': [],
        'tb_writers': {'train_loss': utils.AutoStep(writer.add_scalar, 'client/%s/train_loss'%device_id),
                       'train_acc': utils.AutoStep(writer.add_scalar, 'client/%s/train_acc'%device_id),
                       'test_loss': utils.AutoStep(writer.add_scalar, 'client/%s/test_loss'%device_id),
                       'test_acc' : utils.AutoStep(writer.add_scalar, 'client/%s/test_acc'%device_id)}
        }
  
def train(epoch, device, tb=True):
    device['net'].train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(device['dataloader']):
        inputs, targets = inputs.cuda(), targets.cuda()
        device['optimizer'].zero_grad()
        outputs = device['net'](inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        device['optimizer'].step()
        train_loss += loss.item()
        device['train_loss_tracker'].append(loss.item())
        loss = train_loss / (batch_idx + 1)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        dev_id = device['id']
        # print(f'\r(Device {dev_id}/Epoch {epoch}) ' + 
        #                  f'Train Loss: {loss:.3f} | Train Acc: {acc:.3f}')
    if tb:
        device['train_acc_tracker'].append(acc)
        device['tb_writers']['train_loss'].write(loss)
        device['tb_writers']['train_acc'].write(acc)
    return loss, acc

def test(epoch, device, tb=True):
    device['net'].eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = device['net'](inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            device['test_loss_tracker'].append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            loss = test_loss / (batch_idx + 1)
            acc = 100.* correct / total
    # print(f' | Test Loss: {loss:.3f} | Test Acc: {acc:.3f}\n')
    acc = 100.*correct/total

    if tb:
        device['test_acc_tracker'].append(acc)
        device['tb_writers']['test_loss'].write(loss)
        device['tb_writers']['test_acc'].write(acc)
    return loss, acc
    

def get_devices_for_round(devices, device_pct):
    '''
    '''
    assert device_pct>0 and device_pct<=1, 'device pct must be in the range of (0,1].'
    num_devices_in_round = round(device_pct*len(devices))
    device_idxs = np.random.permutation(len(devices))[:num_devices_in_round]
    return [devices[i] for i in device_idxs]

def update_bn_stat(model, all_inv_image, invbn_epochs=50):
    all_inv_image_tensor = torch.cat(all_inv_image)
    model.train()
    with torch.no_grad():
        for epoch_idx in range(invbn_epochs):
            outputs = model(all_inv_image_tensor)

def update_bn_ondataset(model, dataloader, update_epochs=50):
    model.train()
    with torch.no_grad():
        for epoch_idx in range(update_epochs):
            for inputs, targets in dataloader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)


# def update_weights(model, all_inv_image, all_inv_target, global_lr=0.01, invwei_epochs=50):
#     all_inv_image_tensor = torch.cat(all_inv_image)
#     all_inv_target_tensor = torch.cat(all_inv_target)
#     inv_loader = DataLoader(TensorDataset(all_inv_image_tensor.detach().cpu(), all_inv_target_tensor.detach().cpu()),
#                     batch_size=256, shuffle=True, num_workers=4)
#     optimizer = torch.optim.Adam(model.parameters(), lr=global_lr, weight_decay=5e-4)
#     model.train()

#     for epoch_idx in range(invwei_epochs):
#         train_loss = 0.0
#         total = 0.0
#         correct = 0.0 
#         for batch_idx, (inputs, targets) in enumerate(inv_loader):
#             inputs, targets = inputs.cuda(), targets.cuda()
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#     return train_loss/(batch_idx + 1), 100.*correct/total

# net = model.ConvNet().cuda()
# net = model.CifarNet().cuda()
net = resnet_cifar.ResNet18(num_classes=10).cuda()
criterion = nn.CrossEntropyLoss()

central_device = create_device(net, -1, trainset, list(range(len(trainset))), 
                    lr=0.01)


if args.non_iid:
    data_idxs_dict = fl_data.non_iid_split(trainset, args.num_devices, 
                        shards_per_client=2)
else:
    data_idxs_dict = fl_data.uniform_random_split(trainset, args.num_devices)
# deep copy net for each devices
devices = [create_device(net, i, trainset, data_idxs_dict[i], lr=args.local_lr, batch_size=args.local_bsz)
           for i in range(args.num_devices)]

#########################
## register hooks, inputs and optimizer
#############################
import deepinversion_cifar10
for device in devices+[central_device]:
    loss_r_feature_layers = []
    for module in device['net'].modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(
                deepinversion_cifar10.DeepInversionFeatureHook(module))
    device['loss_r_feature_layers'] = loss_r_feature_layers

    device['di_inputs'] = torch.randn((args.di_batch_size, 3, 32, 32), 
            requires_grad=True, device='cuda')
    device['di_optimizer'] = optim.Adam([device['di_input']], lr=0.05)


start_time = time.time()
for round_num in range(args.rounds):
    round_devices = get_devices_for_round(devices, args.device_pct)
    all_inv_image = []
    all_inv_target = []
    
    for round_device_idx, device in enumerate(round_devices):
        for local_epoch in range(args.local_epochs):
            local_loss, local_acc = train(local_epoch, device)
        logx.msg(f'\r(Device {round_device_idx}) ' + 
                        f'Train Loss: {local_loss:.3f} | Train Acc: {local_acc:.3f}')
        ########################### 
        # do di on local model
        ###########################
        targets = torch.LongTensor(np.random.choice(list(set(device['dataloader'].dataset.targets)), 
                                        args.di_batch_size, replace=True)).to('cuda')
        di_tensor = deepinversion_cifar10.get_images(device['net'], device['loss_r_feature_layers'],
            bs=args.di_batch_size, epochs=2000, idx=-1, var_scale=2.5e-5,
            net_student=None, prefix=None, competitive_scale=0.0, 
            train_writer = None, global_iteration=None,
            use_amp=False,
            optimizer = device['di_optimizer'], inputs = device['di_inputs'], targets = targets,
            bn_reg_scale = 10.0, random_labels = True, l2_coeff=0.0, 
            name_use=logx.logdir, save_image=False)
        device['inv_image'] = targets
        device['inv_target'] = di_tensor
        all_inv_image.append(di_tensor)
        all_inv_target.append(targets)
        
    w_avg = agg.average_weights(round_devices) # average all in the state_dict
    if args.di_bn:
        logx.msg('====> update central device with local di images...')
        central_device['net'].load_state_dict(w_avg)
        update_bn_stat(central_device['net'], all_inv_image, invbn_epochs=50)
        w_avg = central_device['net'].state_dict()

    if args.central_di:
        logx.msg('====> get di image of central device...')
        
    for device in devices:
        device['net'].load_state_dict(w_avg)
        device['optimizer'].zero_grad()
        device['optimizer'].step()
        device['scheduler'].step()

    # test accuracy after aggregation
    round_loss, round_acc = test(round_num, devices[0], tb=False)
    writer.add_scalar('round/loss', round_loss, round_num)
    writer.add_scalar('round/acc', round_acc, round_num)
    print('====> Round:%d, Acc:%.4f'%(round_num, round_acc))


total_time = time.time() - start_time
print('Total training time: {} seconds'.format(total_time))ound_acc, round_num)
    print('====> Round:%d, Acc:%.4f'%(round_num, round_acc))


total_time = time.time() - start_time
print('Total training time: {} seconds'.format(total_time)))