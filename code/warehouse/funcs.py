'''
    funcs.py
'''
import torch
from collections import OrderedDict
import time

# def get_dataloader():
#     from torchvision import datasets, transforms
#     trans_mnist = transforms.Compose([transforms.ToTensor()])
#     dataset_train = datasets.MNIST(
#         './data/MNIST/', train=True, download=False, transform=trans_mnist)
#     dataset_test = datasets.MNIST(
#         './data/MNIST/', train=False, download=False, transform=trans_mnist)
#     return dataset_train, dataset_test

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def clone_state_dict(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        new_state_dict[key] = state_dict[key].clone()
    return new_state_dict

def state_dict_tonumpy(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        new_state_dict[key] = state_dict[key].numpy() if state_dict[key].device==torch.device('cpu') else state_dict[key].cpu().numpy()
    return new_state_dict

def state_dict_fromnumpy(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        new_state_dict[key] = torch.from_numpy(state_dict[key])
    return new_state_dict

def move_to_device(state_dict, target_device):
    new_state_dict = {}
    for key in state_dict.keys():
        new_state_dict[key] = state_dict[key].to(target_device)
    return new_state_dict 

def state_dict_inplace_update(old_state_dict, new_state_dict):
    for key in old_state_dict.keys():
        old_state_dict[key].data.copy_(new_state_dict[key].data.to(old_state_dict[key].device))




def launch_process_update_partial(local_model_queue, global_model, done):
    # counter = 0
    while True:   # scan the queue
        # print('running ...')
        # if not local_model_queue.empty():  
        if True:
            # print('*** Feteching...')
            local_model = local_model_queue.get(block=True)            # get a trained local model from the queue
            # print('**Get one trained model')
            flag = global_model.Incre_FedAvg(local_model)  # add it to partial model
            # counter += 1
            # print('=======', counter)
            # flag = 0
            del local_model
            if flag == 1:
                # done.set()                                               # if enough number of local models are added to partial model
                break                                                   # this process can be shut down
        else: 
            print('*** Local model queue empty...')
            # done.set()
            time.sleep(0.1)                                               # if the queue is empty, keep scaning

def gpu_update_users(user_list, gpu_list):
    '''
    Assign users to each gpu.
    
    Args:
        user_list (list): list of users's index to be assigned
        gpu_list (list:GPUContainer): list of GPUContainer

    Returns:
        gpu_list (list:GPUContainer): the updated gpu_list
    '''
    coordinator = clients_coordinator(clients_list = user_list, 
                    num_of_gpus = len(gpu_list))
    for gpu_idx, users in coordinator.items():
        gpu_list[gpu_idx].update_users(users)  

    return gpu_list








def clients_coordinator(clients_list, num_of_gpus):
    '''
    Args: 
        clients_list (list): list of clients' index to train
        num_of_gpus (int): how many gpus we can use to train

    Returns:
        Dict: key is index of gpu, value is clients' index for this gpu.
    '''
    coordinator = {}
    splited_clients_list = chunkIt(clients_list, num_of_gpus)
    for i in range(num_of_gpus):
        coordinator[i] = splited_clients_list[i]
    return coordinator


def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out


def save_checkpoint(model, filename='checkpoint.pth', optimizer=None, epoch=1):
    # model or model_state
    out = model if type(model) is OrderedDict else model.state_dict()
    out = move_to_device(out, torch.device('cpu'))
    if optimizer is None:
        torch.save({
            'epoch': epoch,
            'state_dict': out,
        }, filename)
    else:
        torch.save({
            'epoch': epoch,
            'state_dict': out,
            'optimizer' : optimizer.state_dict()
        }, filename)


  
