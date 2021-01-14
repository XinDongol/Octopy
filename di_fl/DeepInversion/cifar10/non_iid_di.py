import time
import copy
from copy import deepcopy
from collections import OrderedDict
from argparse import ArgumentParser
import json
import random 
import sys
# sys.path.append('../../../')

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
import agg
import utils
import resnet_cifar
from FedProx import FedProx

from runx.logx import logx
import timm

parser = ArgumentParser()
parser.add_argument('--logdir', type=str)
parser.add_argument('--dataset', type=str, default='cifar10', 
                                choices=['cifar10', 'cifar100', 'svhn'])
parser.add_argument('--rounds', type=int)
parser.add_argument('--num_devices', type=int)
parser.add_argument('--device_pct', type=float)
parser.add_argument('--non_iid', type=int, choices=[0, 1])
parser.add_argument('--split_method', type=int, default=1)
parser.add_argument('--scheduler', type=str, choices=['multistep', 'cosine'])
# local setting
parser.add_argument('--local_epochs', type=int)
parser.add_argument('--local_lr', type=float)
parser.add_argument('--local_bsz', type=int)
parser.add_argument('--local_reset_optim', type=int)

parser.add_argument('--ensure_upload_before', type=int, default=100)
parser.add_argument('--upload_every', type=int, default=1)
# local di process
parser.add_argument('--di_lr', type=float, default=0.05)
parser.add_argument('--di_steps', type=int, default=2000)
parser.add_argument('--di_scheduler', type=int)

parser.add_argument('--local_di', type=int, choices=[0,1])
parser.add_argument('--local_bn_stat_epochs', type=int)
parser.add_argument('--local_di_reset_optim', type=int, choices=[0,1])
parser.add_argument('--local_di_batch_size', type=int)
parser.add_argument('--local_di_celoss', type=float, default=1.0)
parser.add_argument('--local_di_bnloss', type=float, default=10.0)
parser.add_argument('--central_bn_update_celoss', type=float, default=1.0)
parser.add_argument('--central_bn_update_lr', type=float, default=1e-4)
parser.add_argument('--central_bn_update_momentum', type=float, default=0.9)
parser.add_argument('--central_bn_update_epochs', type=int, default=80)
parser.add_argument('--central_bn_update_scheduler', type=int, default=1)

# central di process
parser.add_argument('--central_di', type=int, choices=[0,1,2,3,4])
parser.add_argument('--central_di_celoss', type=float, default=1.0)
parser.add_argument('--central_di_bnloss', type=float, default=10.0)
parser.add_argument('--central_di_reset_optim', type=int, choices=[0,1])
parser.add_argument('--central_di_distill_temp', type=float)
parser.add_argument('--central_di_distill_loss_scale', type=float)
parser.add_argument('--central_di_batch_size', type=int)
parser.add_argument('--local_mix_bsz', type=int)

parser.add_argument('--save_all', type=int, default=0)

# FedProx
parser.add_argument('--fedprox', type=int, choices=[0,1])
parser.add_argument('--fedprox_mu', type=float, default=0.0)
parser.add_argument('--reset_momentum', type=int, choices=[0,1])

args = parser.parse_args()
assert args.central_di_batch_size%args.local_mix_bsz==0

logx.initialize(logdir=args.logdir, coolname=True, tensorboard=True,
                hparams=vars(args))

utils.mkdir(logx.logdir+'/saved')

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
trainset = torchvision.datasets.CIFAR10(root='~/results/cifar', train=True, 
                                        download  = True,
                                        transform = transform_train)

transform_test = transforms.Compose([                                           
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='~/results/cifar', train=False,
                                       download  = True,
                                       transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False,
                                         num_workers=2)



def create_device(args, net, device_id, trainset, data_idxs,
                  milestones=None):
    if milestones == None:
       milestones = [25, 50, 65]

    device_net = copy.deepcopy(net)
    if args.fedprox!=0: 
        optimizer = FedProx(device_net.parameters(), lr=args.local_lr, momentum=0.9,
                                    weight_decay=5e-4, mu=args.fedprox_mu)
    else: 
        # assert fedprox_mu is None
        optimizer = torch.optim.SGD(device_net.parameters(), lr=args.local_lr, momentum=0.9,
                                    weight_decay=5e-4)
    #  optimizer       = torch.optim.Adam(device_net.parameters(), lr=lr, weight_decay=5e-4)
    if args.scheduler== 'multistep':
       scheduler       = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones = milestones,
                                                         gamma      = 0.1)
    else: 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.rounds)
        
    device_trainset    = fl_data.DatasetSplit(trainset, data_idxs)
    device_trainloader = torch.utils.data.DataLoader(device_trainset,
                                                     batch_size  = args.local_bsz,
                                                     shuffle     = True,
                                                     num_workers = 8,          
                                                     drop_last = False)
    return {
        'net'               : device_net,
        'id'                : device_id,
        'dataloader'        : device_trainloader,
        'optimizer'         : optimizer,
        'empty_optim_state_dict': deepcopy(optimizer.state_dict()),
        'scheduler'         : scheduler,
        'train_loss_tracker': [],
        'train_acc_tracker' : [],
        'test_loss_tracker' : [],
        'test_acc_tracker'  : [],
        'tb_writers'        : {'train_loss': utils.AutoStep(writer.add_scalar, 'client/%s/train_loss'%device_id),
                                'train_acc': utils.AutoStep(writer.add_scalar, 'client/%s/train_acc'%device_id),
                                'test_loss': utils.AutoStep(writer.add_scalar, 'client/%s/test_loss'%device_id),
                                'test_acc' : utils.AutoStep(writer.add_scalar, 'client/%s/test_acc'%device_id)},
        'di_image': None,
        'di_target': None
        }
  
def train(args, epoch, device, w_avg, tb=True): 
    device['net'].train()
    train_loss, correct, total = 0, 0, 0
    dataloader = device['dataloader']

    for batch_idx, (inputs, targets) in enumerate(dataloader): 
        inputs, targets = inputs.cuda(), targets.cuda()
        device['optimizer'].zero_grad()
        outputs = device['net'](inputs)
        loss    = criterion(outputs, targets)
        if args.fedprox_mu>0:
            fedprox_reg = 0.0
            for n,p in device['net'].named_parameters():
                assert p.requires_grad==True
                fedprox_reg += ((args.fedprox_mu / 2) * torch.norm((p - w_avg[n].detach()))**2)
            loss += fedprox_reg

        loss.backward()
        device['optimizer'].step()
        train_loss += loss.item()
        device['train_loss_tracker'].append(loss.item())
        loss        = train_loss / (batch_idx + 1)
        _, predicted  = outputs.max(1)
        total      += targets.size(0)
        correct    += predicted.eq(targets).sum().item()
        acc         = 100. * correct / total
        dev_id      = device['id']
        # print(f'\r(Device {dev_id}/Epoch {epoch}) ' + 
        #                  f'Train Loss: {loss:.3f} | Train Acc: {acc:.3f}')
    if tb: 
        device['train_acc_tracker'].append(acc)
        device['tb_writers']['train_loss'].write(loss)
        device['tb_writers']['train_acc'].write(acc)
        
    return loss, acc

def update_local_bn_stat(args, device):
    model = device['net']
    dataloader = device['dataloader']
    model.train()
    with torch.no_grad():
        for e in range(args.local_bn_stat_epochs):
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)


def mixup_data(x_a, y_a, x_b=None, y_b=None, alpha=1.0, use_cuda=True): 
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0: 
        lam = np.random.beta(alpha, alpha)
    else: 
        lam = 1

    if x_b is None: 
        # self mixup
        batch_size = x_a.size()[0]
        if use_cuda: 
            index = torch.randperm(batch_size).cuda()
        else: 
            index = torch.randperm(batch_size)

            mixed_x = lam * x_a + (1 - lam) * x_a[index, :]
        y_a, y_b    = y_a, y_a[index]
    else: 
        mixed_x = lam * x_a + (1 - lam) * x_b
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam): 
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mix_train(epoch, device, central_device, args, tb=True): 
    '''
    mix_mode: 
        1 -- local + cen
        2 -- mix(local, cen)
        3 -- local + mix(cen)
        4 -- local + distill on cen
    '''
    device['net'].train()
    train_loss, correct, total = 0, 0, 0
    all_di_inputs, all_di_targets = central_device['di_image'], central_device['di_target']
    candidate_idx = list(range(all_di_targets.size(0)//args.local_mix_bsz))
    # print(len(central_device['central_di'].dataset))
    # dataloader_iterator = iter(central_device['central_di'])
    
    for batch_idx, (local_inputs, local_targets) in enumerate(device['dataloader']): 
        # device['optimizer'].zero_grad()

        local_inputs, local_targets = local_inputs.cuda(), local_targets.cuda()

        # try: 
        # di_inputs, di_targets = next(dataloader_iterator)
        # except StopIteration: 
        #         dataloader_iterator = iter(central_device['central_di'])
        #         di_inputs,           di_targets        = next(dataloader_iterator)
        selected_idx = int(np.random.choice(candidate_idx))
        di_inputs    = all_di_inputs[selected_idx*args.local_mix_bsz:(selected_idx+1)*args.local_mix_bsz]
        di_targets   = all_di_targets[selected_idx*args.local_mix_bsz:(selected_idx+1)*args.local_mix_bsz]
        di_inputs, di_targets  = di_inputs.cuda(), di_targets.cuda()
        # print('haha', di_inputs.requires_grad, di_targets.requires_grad)

        if args.central_di==1:
            device['optimizer'].zero_grad()
            inputs  = torch.cat([local_inputs, di_inputs])
            targets = torch.cat([local_targets, di_targets])
            outputs = device['net'](inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            device['optimizer'].step()
            
            device['train_loss_tracker'].append([loss.item()])
            _, predicted = outputs[:local_targets.size(0)].max(1)
            total += local_targets.size(0)
            correct += predicted.eq(local_targets).sum().item()
        elif args.central_di==2:
            device['optimizer'].zero_grad()
            mixed_inputs, y_a, y_b, lam = mixup_data(
                local_inputs, local_targets, di_inputs, di_targets,
                alpha = 1.0, use_cuda = True
            ) 
            outputs = device['net'](mixed_inputs)
            loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            device['optimizer'].step()      
            device['train_loss_tracker'].append([loss.item()])      
        elif args.central_di==3:
            device['optimizer'].zero_grad()
            mixed_di_inputs, y_a, y_b, lam = mixup_data(
                di_inputs, di_targets, None, None, 
                alpha=1.0, use_cuda=True)
            inputs     = torch.cat([local_inputs, mixed_di_inputs])
            outputs    = device['net'](inputs)
            part1_loss = criterion(outputs[:local_targets.size(0)], local_targets)
            part2_loss = mixup_criterion(criterion, outputs[local_targets.size(0):], y_a, y_b, lam)
            loss       = (part1_loss + part2_loss) * 0.5
            loss.backward()
            
            device['optimizer'].step()            
            device['train_loss_tracker'].append([part1_loss.item(), part2_loss.item()])
            _, predicted = outputs[:local_targets.size(0)].max(1)
            total += local_targets.size(0)
            correct += predicted.eq(local_targets).sum().item()
        elif args.central_di==4:
            # print('local mix batch idx', batch_idx)
            device['optimizer'].zero_grad()
            
            di_outputs = device['net'](di_inputs)
            s_o     = F.log_softmax(di_outputs/args.central_di_distill_temp, dim=1)
            t_o     = F.log_softmax(di_targets/args.central_di_distill_temp, dim=1)
            loss_kd = F.kl_div(input=s_o, target=t_o, log_target=True) * \
                (args.central_di_distill_temp**2) * \
                args.central_di_distill_loss_scale
            loss_kd.backward()
            # print('di data grad norm', [i.grad.data.flatten().abs().mean().item() for i in optimizer.param_groups[0]['params'][:2]])
            
            outputs = device['net'](local_inputs)
            loss_local = criterion(outputs, local_targets)
            loss_local.backward()            
            
            device['optimizer'].step()    
            
            device['train_loss_tracker'].append([loss_kd.item(), loss_local.item()])      
            _, predicted = outputs.max(1)
            total += local_targets.size(0)
            correct += predicted.eq(local_targets).sum().item()  
        else: 
            raise NotImplementedError

    return train_loss, 100.*correct/total


def test(epoch, device, tb=True): 
    device['net'].eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad(): 
        for  batch_idx, (inputs, targets) in enumerate(testloader): 
            inputs, targets   = inputs.cuda(), targets.cuda()
            outputs    = device['net'](inputs)
            loss       = criterion(outputs, targets)
            test_loss += loss.item()
            device['test_loss_tracker'].append(loss.item())
            _, predicted  = outputs.max(1)
            total      += targets.size(0)
            correct    += predicted.eq(targets).sum().item()
            loss        = test_loss / (batch_idx + 1)
            # acc         = 100.* correct / total
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
    assert device_pct>0 and device_pct<= 1, 'device pct must be in the range of (0,1].'
    num_devices_in_round         = round(device_pct*len(devices))
    device_idxs                  = np.random.permutation(len(devices))[:num_devices_in_round]
    return [devices[i] for i in device_idxs]

def update_bn_stat(central_device, 
                   all_inv_image_tensor, 
                   all_inv_target_tensor, 
                   args, 
                   logx):
    model                 = central_device['net']
    criterion             = nn.CrossEntropyLoss()
    best_acc              = 0.0
    best_state_dict       = None
    
    if args.central_bn_update_celoss>0.0: 
        optimizer = optim.SGD(central_device['net'].parameters(), 
                              lr       = args.central_bn_update_lr,
                              momentum = args.central_bn_update_momentum)
        if args.central_bn_update_scheduler==1:
            #milestones = [args.central_bn_update_epochs//4, args.central_bn_update_epochs//4, ]
            scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                args.central_bn_update_epochs)
        
    test_loss, test_acc = test(0, central_device, tb=False)
    logx.msg('Before BN Update | Acc %f'%(test_acc))
    best_acc = test_acc
    best_state_dict = deepcopy(model.state_dict())
    
    for epoch_idx in range(args.central_bn_update_epochs): 
        model.train()
        model.zero_grad()
        output = model(all_inv_image_tensor)
        if args.central_bn_update_celoss>0.0: 
            loss = criterion(output, all_inv_target_tensor)
            loss.backward()
            optimizer.step()
            
            if args.central_bn_update_scheduler: 
                scheduler.step()
            
        if epoch_idx%5==0:
            test_loss, test_acc = test(0, central_device, tb=False)
            logx.msg('During BN Update %d | Acc %f'%(epoch_idx, test_acc))
            if test_acc>best_acc: 
                best_acc = max(test_acc, best_acc)
                # del best_state_dict
                best_state_dict = deepcopy(model.state_dict())
                
    model.load_state_dict(best_state_dict)
    test_loss, test_acc = test(0, central_device, tb=False)
    logx.msg('After BN Update | Acc %f'%(test_acc))
    # del best_state_dict


# net       = model.ConvNet().cuda()
# net       = model.CifarNet().cuda()
net       = resnet_cifar.ResNet18(num_classes=10).cuda()
criterion = nn.CrossEntropyLoss()

central_device = create_device(args=args, 
                               net=net, 
                               device_id=-1, 
                               trainset=trainset, 
                               data_idxs=list(range(len(trainset))), 
                               )


if args.non_iid: 
    if args.split_method==1:
        data_idxs_dict = fl_data.non_iid_split(trainset, args.num_devices, 
                        shards_per_client=2)
    elif args.split_method==2:
        data_idxs_dict = fl_data.dir_split(trainset, args.num_devices, 
                        alpha=0.5) 
    else:
        raise NotImplementedError       
else: 
    data_idxs_dict = fl_data.uniform_random_split(trainset, args.num_devices)
# deep copy net for each devices
devices = [
    create_device(args=args,
                  net=net, 
                  device_id=i, 
                  trainset=trainset, 
                  data_idxs=data_idxs_dict[i]) 
           for i in range(args.num_devices)
           ]
if args.save_all:
    np.save(logx.logdir+'/data_part.npy', data_idxs_dict)
#########################
## register hooks, inputs and optimizer
#############################
import deepinversion_cifar10

for d_idx, device in enumerate([central_device]+devices): 
    loss_r_feature_layers = []
    for module in device['net'].modules() : 
        if  isinstance(module, nn.BatchNorm2d): 
                loss_r_feature_layers.append(
                    deepinversion_cifar10.DeepInversionFeatureHook(module))
    device['loss_r_feature_layers'] = loss_r_feature_layers

    if d_idx==0:
        device['di_inputs'] = torch.randn((args.local_mix_bsz, 3, 32, 32), 
                requires_grad=True, device='cuda')
    else: 
        device['di_inputs'] = torch.randn((args.local_di_batch_size, 3, 32, 32), 
                requires_grad=True, device='cuda')        
    device['di_optimizer'] = optim.Adam([device['di_inputs']], lr=args.di_lr)
    
##############################
# start FL
##############################
w_avg = central_device['net'].state_dict()

start_time = time.time()
for round_num in range(args.rounds): 
    round_devices  = get_devices_for_round(devices, args.device_pct)
    all_inv_image  = []
    all_inv_target = []
    
    for round_device_idx, device in enumerate(round_devices): 
        for local_epoch in range(args.local_epochs): 
            if  (args.central_di==0) or (round_num==0) : # no central_di
                if  args.fedprox==0: 
                    local_loss, local_acc = train(args, local_epoch, device, w_avg)
                elif  args.fedprox==1:
                    # update optimizer
                    device['optimizer'].update_old_init(args.reset_momentum==1)
                    local_loss, local_acc = train(args, local_epoch, device, w_avg)
                else: 
                    raise NotImplementedError
            elif args.central_di in [1,2,3,4]: 
                local_loss, local_acc = mix_train(local_epoch, device, central_device, args,
                    tb=True)
            else: 
                raise NotImplementedError

        logx.msg(f'\r(Device {round_device_idx}) ' + 
                        f'Train Loss: {local_loss:.3f} | Train Acc: {local_acc:.3f}')
        # just for debugging
        if args.save_all:
            test_loss, test_acc = test(round_num, device, tb=True)
            logx.msg(f'\r(Device {round_device_idx}) ' + 
                        f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}')
            
        
        if args.local_bn_stat_epochs>0:
            logx.msg('Updating bn stats')
            update_local_bn_stat(args, device)
        
        if args.local_di: 
            ########################### 
            # do di on local model
            ###########################
            targets = torch.LongTensor(np.random.choice(list(set(device['dataloader'].dataset.targets)), 
                                            args.local_di_batch_size, replace=True)).to('cuda')
            di_tensor, _ = deepinversion_cifar10.get_images(device['net'], 
                                                            device['loss_r_feature_layers'],
                                                            args,
                                                            bs                = args.local_di_batch_size,
                                                            epochs            = args.di_steps,
                                                            idx               = -1,
                                                            var_scale         = 2.5e-5,
                                                            net_student       = None,
                                                            prefix            = None,
                                                            competitive_scale = 0.0,
                                                            train_writer      = None,
                                                            global_iteration  = None,
                                                            use_amp           = False,
                                                            main_loss         = args.local_di_celoss,
                                                            optimizer         = None if args.local_di_reset_optim else device['di_optimizer'],
                                                            inputs            = device['di_inputs'],
                                                            targets           = targets,
                                                            bn_reg_scale      = args.local_di_bnloss,
                                                            random_labels     = True,
                                                            l2_coeff          = 0.0,
                                                            name_use          = logx.logdir,
                                                            save_image        = False)
            del device['di_image'], device['di_target']
            device['di_image']  = di_tensor
            device['di_target'] = targets
            all_inv_image.append(di_tensor)
            all_inv_target.append(targets)
            
    if args.save_all:
        save_dict = {str(d['id'])+'_model':d['net'].state_dict() for d in round_devices}
        save_dict.update({str(d['id'])+'_diimage':d['di_image'] for d in round_devices})
        save_dict.update({str(d['id'])+'_ditarget':d['di_target'] for d in round_devices})
        
    if (round_num>args.ensure_upload_before \
        and (round_num-args.ensure_upload_before)%args.upload_every==0) \
            or (round_num<=args.ensure_upload_before):
        # weight average
        logx.msg('%d Round upload local model'%round_num)
        w_avg = agg.average_weights(round_devices) # average all in the state_dict
    else:
        logx.msg('%d Round NOT upload local model'%round_num)

    if args.save_all:
        central_sd_before_update = deepcopy(w_avg)
    # local di
    if args.local_di: 
        logx.msg('====> Round %d| update central device with local di images...'%round_num)
        central_device['net'].load_state_dict(w_avg)
        all_inv_image_tensor, all_inv_target_tensor = torch.cat(all_inv_image), torch.cat(all_inv_target)
        
        # if 1:
        #     torch.save({'image': all_inv_image_tensor, 
        #                 'target': all_inv_target_tensor}, logx.logdir+'/%d-local_di.pth'%round_num)
        update_bn_stat(central_device, all_inv_image_tensor, all_inv_target_tensor, args, logx)
        del all_inv_image_tensor, all_inv_target_tensor, all_inv_image, all_inv_target
        w_avg = central_device['net'].state_dict()

    # central di
    if args.central_di!= 0:
        
        logx.msg('====> get di image of central device...')
        central_device['net'].load_state_dict(w_avg)
        # central_di_num = int(len(data_idxs_dict[0])/args.di_batch_size + 1)
        central_di_num = args.central_di_batch_size//args.local_mix_bsz
        
        all_central_di_image  = []
        all_central_di_target = []
        for central_di_idx in range(central_di_num): 
            
            if args.central_di==4:
                targets = None
            else:
                targets = torch.LongTensor(np.random.choice(10, 
                                    args.local_mix_bsz, replace=True)).to('cuda')
            di_image, _ = deepinversion_cifar10.get_images(central_device['net'],
                                                           central_device['loss_r_feature_layers'],
                                                           args,
                                                            bs                = args.local_mix_bsz,
                                                            epochs            = args.di_steps,
                                                            idx               = -1,
                                                            var_scale         = 2.5e-5,
                                                            net_student       = None,
                                                            prefix            = None,
                                                            competitive_scale = 0.0,
                                                            train_writer      = None,
                                                            global_iteration  = None,
                                                            use_amp           = False,
                                                            main_loss         = args.central_di_celoss,
                                                            optimizer         = None if args.central_di_reset_optim else central_device['di_optimizer'],
                                                            inputs            = central_device['di_inputs'],
                                                            targets           = targets,
                                                            bn_reg_scale      = args.central_di_bnloss,
                                                            random_labels     = True,
                                                            l2_coeff          = 0.0,
                                                            name_use          = logx.logdir,
                                                            save_image        = False)
            
            all_central_di_image.append(di_image)
            if args.central_di==4:
                targets = central_device['net'](di_image)
            all_central_di_target.append(targets.data)
        
        # del central_device['di_image'], central_device['di_target']
        if central_device['di_image'] is not None:
            central_device['di_image'].copy_(torch.cat(all_central_di_image))
            central_device['di_target'].copy_(torch.cat(all_central_di_target))
        else:
            central_device['di_image']=torch.cat(all_central_di_image)
            central_device['di_target']=torch.cat(all_central_di_target)       
        del all_central_di_image, all_central_di_target
        
    if args.save_all:
        save_dict.update({'central_diimage': central_device['di_image'],
                          'central_ditarget': central_device['di_target'],
                          'w_avg_before_update': central_sd_before_update,
                          'w_avg_after_update': w_avg})
        torch.save(save_dict, logx.logdir+'/saved/%d.pth.tar'%round_num)
        
    for device in devices: 
        device['net'].load_state_dict(w_avg)
        # device['optimizer'].zero_grad()
        # device['optimizer'].step()
        device['scheduler'].step()
        if args.local_reset_optim==1:
            device['optimizer'].load_state_dict(device['empty_optim_state_dict'])


    # test accuracy after aggregation
    round_loss, round_acc = test(round_num, devices[0], tb=False)
    metrics    = {'top1': round_acc, 'loss': round_loss}
    logx.metric('val', metrics, round_num)
    logx.msg('====> Round:%d, Acc:%.4f'%(round_num, round_acc))


total_time = time.time() - start_time
logx.msg('Total training time: {} seconds'.format(total_time)) # test accuracy after aggregation