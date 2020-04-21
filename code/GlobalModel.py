from copy import deepcopy
import torch
from torch import nn
from warehouse.funcs import *
import numpy as np
import pickle


class Global_Model:
    def __init__(self, model, capacity):
        '''
        capacity = num of gpus
        '''
        self.incre_counter = 0      # within current round, how many local models you have received
        self.round = 0              # current round of Federated Learning
        self.capacity = capacity    # how many local models you expect to receive per round
        self.saved_state_dict = init_saved_state_dict(model)
        self.saved_model = model  # global model of last round
        self.received_models = []

    '''
    def Incre_FedAvg(self, w_in):
        self.incre_counter += 1
        
        if self.incre_counter == 1:
            for k in self.__state_dict.keys():
                self.__state_dict[k] = w_in[k] / self.capacity
            return 0

        for k in self.__state_dict.keys():  # iterate every weight element
            self.__state_dict[k] += w_in[k] / self.capacity

        if self.incre_counter == self.capacity:
            self.round += 1
            self.incre_counter = 0
            self.saved_model.load_state_dict(
                state_dict_fromnumpy(self.__state_dict))
            return 1

        return 0
    '''
    
    def Incre_FedAvg(self, w_in):
        self.received_models.append(w_in)
        if len(self.received_models) == self.capacity:
            # ------v
            # pickle.dump(self.received_models, open('./test_save/received_models.p', 'wb'))
            # ------^
            self.round += 1
            self.saved_state_dict = average_weights(self.received_models)
            # ------v
            # pickle.dump(self.saved_state_dict, open('./test_save/saved_state_dict.p', 'wb'))
            # assert 1==0
            # ------^
            self.saved_model.load_state_dict(
                state_dict_fromnumpy(self.saved_state_dict))
            self.received_models = []
            return 1
        else:
            return 0
            
            
        

    def evaluate(self, val_dataset):
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=100, shuffle=False,
                                                 num_workers=0)

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        # switch to evaluate mode
        self.saved_model.eval()
        # TODO: this should be got from config
        # criterion = nn.CrossEntropyLoss().cuda()
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):

                # images = images.cuda(non_blocking=True)
                # target = target.cuda(non_blocking=True)

                # compute output
                output = self.saved_model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # if i % args.print_freq == 0:
                #     progress.display(i)

            # TODO: this should also be done with the ProgressMeter
            # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            #       .format(top1=top1, top5=top5))

        return top1.avg, losses.avg


def average_weights(w):
    '''
    Returns the average of the weights.
    '''
    w_avg = deepcopy(w[0])
    keys = list(w_avg.keys())
    for key in keys:
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] /=  len(w)
    
    for key in keys:
        w_avg[key+'.var'] = (w[0][key]-w_avg[key]) ** 2
    
    for key in keys:
        for i in range(1, len(w)):
            w_avg[key+'.var'] += (w[i][key]-w_avg[key]) ** 2
        w_avg[key+'.var'] /= len(w)
        
    return w_avg


def init_saved_state_dict(model):
    saved_state_dict = state_dict_tonumpy(
            model.state_dict())  # weights of global model in numpy
    
    for key in model.state_dict().keys():
        saved_state_dict[key+'.var'] = np.ones_like(saved_state_dict[key])
        
    return saved_state_dict