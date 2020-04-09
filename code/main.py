import torch
import torch.multiprocessing as mp

from GlobalModel import Global_Model
# from PartialModel import Partial_Model
from warehouse.funcs import *
from dataset import get_dataset
from GPUContainer import GPU_Container
from Config import BaseConfig
import models
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from copy import deepcopy


def initialize_global_model(config):
    # initialize global model on CPU
    global_net = models.__dict__[config.opt.model]()
    global_model = Global_Model(
        model=global_net, capacity=config.opt.num_users)
    return global_model


def update_user_config(round_idx, config, 
                       user_list, 
                       train_dataset, test_dataset, 
                       splitted_data, 
                       default_user=None):
    '''
    Update the config of users before each round.

    Args:
        user_list (list): index of users
        config (BaseConfig): configuration
        default_user (None or Dict): setting of default user

    Returns:
        config (BaseConfig): the updated configuration
    '''
    user_config = {}

    # get the setting of default user
    if default_user is None:
        default_user = {}
        for group in config.parser._action_groups:
            if group.title == 'user_group':
                group_dict = {a.dest: getattr(
                    config.opt, a.dest, None) for a in group._group_actions}
                default_user.update(argparse.Namespace(**group_dict).__dict__)

    for user in user_list:
        user_config[user] = deepcopy(default_user)
        # you can add more config here
        # For Example:
        # user_config[user]['dir'] = '/home/data'
        user_config[user]['lr'] = config.opt.lr * (0.1**(round_idx//50))
        # print('#######################', user, splitted_data[user])
        user_config[user]['local_train_dataset'] = deepcopy(splitted_data[user])
        # print('***********************', user, user_config[user]['local_train_dataset'])
        user_config[user]['local_test_dataset'] = range(len(test_dataset))
        # print('**********>>>>>>>>>>', user_config)
    return user_config


def main():
    config = BaseConfig()
    config.parse_args(save=True)
    mp.set_start_method('spawn', force=True)

    # initialize global model
    global_model = initialize_global_model(config)

    # split dataset
    train_dataset, test_dataset, splitted_data = get_dataset(config.opt)

    # setup up event for processes
    done = mp.Event()
    # setup queue for trained local models
    # you may have to be careful with maxsize
    local_model_queue = mp.Queue(maxsize=10)
    # setup queue for users to be trained
    user_queue_for_processings = mp.Queue()
    # manager dict
    manager = mp.Manager()
    all_users_opt = manager.dict()
    user_config = update_user_config(round_idx=-1, config=config,
                                    user_list=list(
                                        range(int(config.opt.num_users))),
                                    train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    splitted_data=splitted_data,
                                    default_user=None)
    all_users_opt.update(user_config)

    # setup gpu container for each gpu
    GPU_Containers = []
    for gpu_idx in range(config.opt.num_gpu):
        GPU_Containers.append(GPU_Container(device=torch.device('cuda:'+str(gpu_idx)),
                                            gpu_parallel=config.opt.num_local_models_per_gpu,
                                            all_users_opt=all_users_opt,
                                            global_model=global_model,
                                            local_model_queue=local_model_queue,
                                            user_queue_for_processings=user_queue_for_processings,
                                            model_name=config.opt.model,
                                            train_dataset=train_dataset,
                                            test_dataset=test_dataset))
        # different gpus share the queue
        # the queue is in the cpu

    # lunch processes
    assert len(GPU_Containers) == config.opt.num_gpu
    local_process_list = []
    for gpu_launcher in GPU_Containers:
        # gpu_launcher.update_done(done)   # update event for each round
        local_process_list += gpu_launcher.launch_gpu()
    # ---------------------- Start --------------------------
    writer = SummaryWriter(config.opt.expr_dir)
    test_loss_w = AutoStep(writer.add_scalar, 'test/loss')
    test_acc_w = AutoStep(writer.add_scalar, 'test/acc')
    
    for round_idx in range(config.opt.num_rounds):
        start_time = time.time()
        # update users' configuration
        user_config = update_user_config(round_idx=round_idx, config=config,
                                    user_list=list(
                                        range(int(config.opt.num_users))),
                                    train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    splitted_data=splitted_data,
                                    default_user=None)
        all_users_opt.update(user_config)
        # print('=============>', all_users_opt)
        
        # print('Get user config')
        # start multiprocessing training for each gpu
        for gpu_launcher in GPU_Containers:
            # gpu_launcher.update_done(done)   # update event for each round
            gpu_launcher.update_true_global(global_model)   # pull from global model
        # print('Update user config and true global')

        global_model.capacity = int(config.opt.num_users*config.opt.user_fraction)
        selected_users = list(np.random.choice(config.opt.num_users, 
                                          global_model.capacity,
                                          replace=False))
        
        # print('selected_user', selected_users)
        [user_queue_for_processings.put(i) for i in selected_users]
        # print('---------------------- queue size', user_queue_for_processings.qsize())

        # take trained local models from the queue and then aggregate them into global model
        launch_process_update_partial(local_model_queue, global_model, done)
        print('======> round:', round_idx, abs_mean_state_dict(global_model.saved_state_dict))

        
        test_top1, test_loss = global_model.evaluate(test_dataset)
        test_loss_w.write(test_loss)
        test_acc_w.write(test_top1)
        print('='*40)
        print("Time %.3f, Round %d, Participants %d, LR %e, Loss %.4f, Acc %.4f" % 
              (time.time()-start_time, round_idx, global_model.capacity, all_users_opt[0]['lr'], test_loss, test_top1))
        
    
    writer.close()
    for p in local_process_list:
        p.terminate()
        p.join()

    # save_checkpoint(global_model.saved_model.state_dict(), 'checkpoint_global.pth')


if __name__ == '__main__':
    main()

