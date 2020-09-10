import time
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import model
import fl_data
import quant
import utils
import agg

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./test')
nbit = 32
rounds = 100
local_epochs = 20
num_devices = 5
device_pct = 1.
local_lr = 0.01
global_lr = 0.05

# Using CIFAR-10 again as in Assignment 1
# Load training data
transform_train = transforms.Compose([                                   
    transforms.RandomCrop(32, padding=4),                                       
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='/tmp', train=True, 
                                        download=True,
                                        transform=transform_train)

# Load testing data
transform_test = transforms.Compose([                                           
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='/tmp', train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False,
                                         num_workers=2)


# Using same ConvNet as in Assignment 1


def create_device(net, device_id, trainset, data_idxs, lr=0.1,
                  milestones=None, batch_size=128):
    if milestones == None:
        milestones = [25, 50, 75]

    device_net = copy.deepcopy(net)
    # optimizer = torch.optim.SGD(device_net.parameters(), lr=lr, momentum=0.9,
    #                             weight_decay=5e-4)
    optimizer = torch.optim.Adam(device_net.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=milestones,
                                                     gamma=0.1)
    device_trainset = fl_data.DatasetSplit(trainset, data_idxs)
    device_trainloader = torch.utils.data.DataLoader(device_trainset,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     num_workers=2)
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
    device['train_acc_tracker'].append(acc)

    if tb:
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
    print(f' | Test Loss: {loss:.3f} | Test Acc: {acc:.3f}\n')
    acc = 100.*correct/total
    device['test_acc_tracker'].append(acc)
    
    if tb:
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


# net = model.ConvNet().cuda()
net = model.CifarNet().cuda()
criterion = nn.CrossEntropyLoss()

data_idxs_dict = fl_data.uniform_random_split(trainset, num_devices)
# deep copy net for each devices
devices = [create_device(net, i, trainset, data_idxs_dict[i], lr=local_lr)
           for i in range(num_devices)]

w_avg = net.state_dict()
## IID Federated Learning

start_time = time.time()
for round_num in range(rounds):
    round_devices = get_devices_for_round(devices, device_pct)
    for round_device_idx, device in enumerate(round_devices):
        for local_epoch in range(local_epochs):
            local_loss, local_acc = train(local_epoch, device)
        print(f'\r(Device {round_device_idx}) ' + 
                        f'Train Loss: {local_loss:.3f} | Train Acc: {local_acc:.3f}')
        # quant.quantize_model(device['net'], nbit)
        device['binary_diff'] = quant.sign_state_dict(agg.diff_model(old=w_avg, new=device['net']))

    # new_w_avg = agg.average_weights(round_devices)
    w_binary_diff = agg.majority_vote(round_devices)
    
    # update the old avg
    w_avg = agg.apply_diff(old=w_avg, diff=w_binary_diff, lr=global_lr)

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
print('Total training time: {} seconds'.format(total_time))