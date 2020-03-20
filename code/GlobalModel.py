from copy import deepcopy
import torch
from torch import nn
from warehouse.funcs import *

class Global_Model:
    def __init__(self, model, capacity):
        '''
        capacity = num of gpus
        '''
        self.incre_counter = 0      # within current round, how many local models you have received
                                    # NOTE: before you pull global model, you may want to make sure that incre_conuter == 0
        self.round = 0              # current round of Federated Learning
        self.capacity = capacity    # how many local models you expect to receive per round
        self.state_dict = state_dict_tonumpy(model.state_dict())  # weights of global model
        self.saved_model = model # global model of last round


    def Incre_FedAvg(self, w_in):
        self.incre_counter += 1
        # print('**** Updated global ', self.incre_counter)
        # print('counter: ', self.incre_counter)
        if self.incre_counter == 1:
            for k in self.state_dict.keys():
                self.state_dict[k] = w_in[k] / self.capacity
            # print('flag: ', 0)
            return 0

        for k in self.state_dict.keys():  # iterate every weight element
            self.state_dict[k] += w_in[k] / self.capacity
            # self.state_dict[k] += torch.div(w_in[k].cpu(), self.incre_counter)
            # self.state_dict[k] = torch.div(self.state_dict[k], 1/self.incre_counter+1)
        

        
        if self.incre_counter == self.capacity:
            # print('This is the end of this round ...')
            self.round += 1
            self.incre_counter = 0
            self.saved_model.load_state_dict(state_dict_fromnumpy(self.state_dict))
            # print('flag: ', 1)
            return 1

        # print('flag: ', 0)
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
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

        return top1.avg, losses.avg



        
    
