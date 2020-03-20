# from PartialModel import Partial_Model 
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

    # print("launch local model training process: ", device, processing_index)
    #print('true global', true_global)
    ready_model = models.__dict__[model_name]()
    ready_model.to(device)

    # local_train_idx = []
    # local_test_idx = []
    # train_sampler = SubsetRandomSampler(local_train_idx)

    train_loader = DataLoader(train_dataset, 
                                    batch_size=all_users_opt[0]['local_batchsize'],
                                    sampler=SubsetRandomSampler([]),
                                    # shuffle=True
                                    num_workers=0,
                                    )
    test_loader  = DataLoader(test_dataset, 
                                    sampler=SubsetRandomSampler([]), 
                                    # shuffle=False
                                    num_workers=0,
                                    )

    #print('1')
    while True:
        # if (not user_queue_for_processings.empty()) and local_model_queue.empty():
        # if (not user_queue_for_processings.empty()) :
        if True:
            # done.wait()
            # print("Fetching user: ", device, processing_index)
            user_index = user_queue_for_processings.get(block=True)
            # print('Get user index: -- %d' % user_index)
            #print('2: ', ready_model, '|', true_global['fc2.bias'].device)

            # print("Fetching true global: ", device, processing_index)
            # get and put back global model
            # true_global = true_global_queue.get(block=True)
            # put_back_true_global = clone_state_dict(true_global)
            # true_global_queue.put(put_back_true_global, block=True)
            # print('Try put trained local: * %d' % user_index)

            ready_model.load_state_dict(true_global)
            if user_index in [10,100,150]:
                print('ready_model:', user_index,ready_model.fc1.weight)
            # del true_global
            #print('3')
            current_user = User(user_index=user_index, 
                                ready_model=ready_model, 
                                all_users_opt=all_users_opt, 
                                train_loader=train_loader, 
                                test_loader=test_loader)
            #print('4')
            # print("Trainging ... : ", device, processing_index, user_index)
            current_user.local_train()
            #print('5')
            # trained_state_dict = move_to_device(current_user.net.state_dict(), torch.device('cpu'))
            trained_state_dict = state_dict_tonumpy(current_user.net.state_dict())
            # print('------ Trained', user_index, trained_state_dict['fc1.weight'])
            local_model_queue.put(trained_state_dict, 
                                    block=True)
            # print('Put trained local: # %d' % user_index)
            time.sleep(np.random.random_sample()*0.2+0.1)

        else:
            time.sleep(np.random.random_sample()+0.1)
            print("### User queue is empty ", device, processing_index)
        
    # print("Ending local model training process: ", device, processing_index)
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
        # Use this queue to spread gpu model to different processes
        # self.true_global_queue = mp.Queue(maxsize=1)
        # state dict of the true global model for current round
        self.true_global = move_to_device(global_model.saved_model.state_dict(), self.device)
        for key in self.true_global.keys():
            self.true_global[key] = self.true_global[key].share_memory_()

        self.model_name = model_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.done = None
        

    # def split_for_processings(self):
    #     self.user_list_for_processings = chunkIt(self.users, self.gpu_parallel)
        

    # def update_users(self, users):
    #     self.users = users           # update the uses this phsical gpu device responses for
    #     self.split_for_processings() # then split users into different processes

    def update_done(self, done):
        self.done = done
    
    def update_true_global(self, global_model):
        # pull true_global from the cloud, which is the start point of this round
        # self.global_model = global_model
        # self.true_global = move_to_device(copy.deepcopy(global_model.saved_model.state_dict()), self.device)
        # if not self.true_global_queue.empty():
        #     old_true_global = self.true_global_queue.get(block=True)
        #     del old_true_global
        # # pull new true global model
        # self.true_global_queue.put(move_to_device(global_model.saved_model.state_dict(), self.device), block=True)
        state_dict_inplace_update(self.true_global, global_model.saved_model.state_dict())

        



    def launch_gpu(self):
        # assert self.done is not None

        local_process_list = []  # all processes for this gpu
        for processing_index in range(self.gpu_parallel):
            new_p = mp.Process(target=launch_one_processing, \
                    args=(processing_index, self.true_global, self.device, 
                            self.user_queue_for_processings, self.local_model_queue, 
                            self.all_users_opt, 
                            self.model_name, self.train_dataset, self.test_dataset,
                            self.done))
            new_p.start()
            local_process_list.append(new_p)


        return local_process_list

        

        
