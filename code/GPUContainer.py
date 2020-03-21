from warehouse.funcs import *
from Users import User
import models
import torch.multiprocessing as mp
import copy
import time
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np


def launch_one_processing(processing_index, true_global, device,
                          user_queue_for_processings, local_model_queue,
                          all_users_opt,
                          model_name, train_dataset, test_dataset,
                          done):
    ready_model = models.__dict__[model_name]()
    ready_model.to(device)
    train_loader = DataLoader(train_dataset,
                              batch_size=all_users_opt[0]['local_batchsize'],
                              sampler=SubsetRandomSampler([]),
                              # shuffle=True
                              num_workers=0,
                              )
    test_loader = DataLoader(test_dataset,
                             sampler=SubsetRandomSampler([]),
                             # shuffle=False
                             num_workers=0,
                             )
    while True:
        # done.wait()
        user_index = user_queue_for_processings.get(block=True)
        ready_model.load_state_dict(true_global)
        
        current_user = User(user_index=user_index,
                            ready_model=ready_model,
                            all_users_opt=all_users_opt,
                            train_loader=train_loader,
                            test_loader=test_loader)
        current_user.local_train()
        
        trained_state_dict = state_dict_tonumpy(current_user.net.state_dict())
        local_model_queue.put(trained_state_dict, block=True)
        
        time.sleep(np.random.random_sample()*0.2+0.1)

    # done.wait()
    # print("**Ending local model training process: ", device, processing_index)


class GPU_Container:
    def __init__(self, device, gpu_parallel,
                 all_users_opt,
                 global_model,
                 local_model_queue, user_queue_for_processings,
                 model_name, train_dataset, test_dataset):
        self.all_users_opt = all_users_opt
        self.gpu_parallel = gpu_parallel
        self.device = device            # which phsical gpu device to use
        self.local_model_queue = local_model_queue
        self.user_queue_for_processings = user_queue_for_processings
        # state dict of the true global model for current round
        self.true_global = move_to_device(
            global_model.saved_model.state_dict(), self.device)
        for key in self.true_global.keys():
            self.true_global[key] = self.true_global[key].share_memory_()

        self.model_name = model_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.done = None

    # def update_done(self, done):
    #     self.done = done

    def update_true_global(self, global_model):
        state_dict_inplace_update(
            self.true_global, global_model.saved_model.state_dict())

    def launch_gpu(self):
        # assert self.done is not None
        local_process_list = []  # all processes for this gpu
        for processing_index in range(self.gpu_parallel):
            new_p = mp.Process(target=launch_one_processing,
                               args=(processing_index, self.true_global, self.device,
                                     self.user_queue_for_processings, self.local_model_queue,
                                     self.all_users_opt,
                                     self.model_name, self.train_dataset, self.test_dataset,
                                     self.done))
            new_p.start()
            local_process_list.append(new_p)

        return local_process_list
