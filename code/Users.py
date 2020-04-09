import torch
from torch import nn, autograd

import numpy as np
import random
from torch.autograd import Variable
from torch.utils import data
from warehouse.funcs import save_checkpoint
import torch.nn.functional as F


class User(object):
    def __init__(self, user_index, ready_model, true_global,
                 all_users_opt,
                 train_loader, test_loader):
        
        self.user_opt = all_users_opt[user_index]
        self.user_index = user_index

        self.train_loader, self.test_loader = train_loader, test_loader
        update_list(self.train_loader.sampler.indices,
                    self.user_opt['local_train_dataset'])
        update_list(self.test_loader.sampler.indices,
                    self.user_opt['local_test_dataset'])

        self.net = ready_model
        self.true_global = true_global
        self.device = torch.device(next(self.net.parameters()).device)

    def get_optimizer(self):
        if self.user_opt['optimizer'] == 'SGD':
            # print(self.user_opt['lr'])
            return torch.optim.SGD(self.net.parameters(), lr=self.user_opt['lr'],
                                   weight_decay=self.user_opt['weights_decay'],
                                   momentum=self.user_opt['momentum'])
        elif self.user_opt['optimizer'] == 'Adam':
            return torch.optim.Adam(self.net.parameters(), lr=self.user_opt['lr'], betas=(0.0, 0.99),
                                    weight_decay=self.user_opt['weights_decay'])
        else:
            raise NotImplementedError(
                "Please implement %s in User.get_optimizer" % self.user_opt['optimizer'])

    def get_loss_func(self):
        if self.user_opt['loss_func'] == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(
                "Please implement %s in User.get_optimizer" % self.user_opt['loss_func'])

    def local_train(self):
        optimizer = self.get_optimizer()
        loss_func = self.get_loss_func()
        for tensor in self.true_global.values():
            assert tensor.requires_grad == False
        self.net.train()
        for epoch in range(1, self.user_opt['local_epoch'] + 1):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.net(data)
                baseloss = loss_func(output, target)
                ewcloss = ewc_loss(self.net.named_parameters(), self.true_global, 200)
                # print('base_loss', baseloss)
                # print('ewc_loss', ewcloss)
                loss = baseloss + ewcloss
                
                loss.backward()
                optimizer.step()

        # save_checkpoint(self.net, filename='checkpoint_%d.pth' % self.user_index)

    # def local_train(self):
    #     '''
    #     Used for test.
    #     '''
    #     # print('User', self.user_index, self.net.state_dict())
    #     for epoch in range(1, self.user_opt['local_epoch'] + 1):
    #         for batch_idx, data in enumerate(self.train_loader):
    #             print('User', self.user_index, data)

    #     # fake model update
    #     self.net.fc1.weight.data.add_(self.user_index,
    #                                   torch.ones_like(self.net.fc1.weight.data))


def ewc_loss(current_named_parameters, previous_state_dict, lambda_factor):
    loss = 0
    for key, val in current_named_parameters:
        loss += torch.div( (val - previous_state_dict[key])**2,
                 previous_state_dict[key+'.var'] ).sum()
        
    return 0.5*lambda_factor*loss


def update_list(old_l, new_l):
    '''
    Update the old list (old_l) in-place with new list (new_l)
    '''
    [old_l.pop() for i in range(len(old_l))]
    old_l.extend(new_l)
