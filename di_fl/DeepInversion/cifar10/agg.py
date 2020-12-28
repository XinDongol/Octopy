import torch
from collections import OrderedDict
import torch.nn as nn
from copy import deepcopy


def average_weights(devices):
    '''
    Returns the average of the weights.
    '''
    w = [device['net'].state_dict() for device in devices]
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], float(len(w)))
        
    return w_avg


def diff_model(old, new):
    '''
    take models or their state_dict as input, return the diff of state_dict
    '''
    old_state_dict = old if isinstance(old, OrderedDict) else old.state_dict()
    new_state_dict = new if isinstance(new, OrderedDict) else new.state_dict()
    diff_state_dict = OrderedDict()
    for key in old_state_dict.keys():
        diff_state_dict[key] = new_state_dict[key] - old_state_dict[key]
    return diff_state_dict


def majority_vote(devices):
    w = [device['binary_diff'] for device in devices]
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key].sign_()
        
    return w_avg

def apply_diff(old, diff, lr):
    old = old.state_dict() if isinstance(old, nn.Module) else old
    diff = diff.state_dict() if isinstance(diff, nn.Module) else diff

    for key in old.keys():
        old[key].add_(other=diff[key], alpha=lr)

    return old
