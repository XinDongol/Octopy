# from Config import BaseConfig
# import argparse
# from main import update_user_config

# config = BaseConfig()
# config.parse_args(False)
# config = update_user_config(user_list = list(range(int(config.opt.num_users))), 
#                             config = config, default_user=None)

# print(config.opt.users)

# ------------ test how to share cpu tensor across process --------

# import torch
# import torch.multiprocessing as mp
# import time

# t = torch.tensor([1,2,3])
# t.share_memory_()

# state_dict = {'t': t}

# def test_p(state_dict):
#     for i in range(3):
#         print(state_dict['t'])
#         time.sleep(5)

# p = mp.Process(target=test_p, args=(state_dict,))
# p.start()

# time.sleep(1)
# for k in state_dict.keys():
#     state_dict[k].data.copy_(torch.tensor([4,5,6]))
# # you can NOT do: t = torch.tensor([4,5,6]) because it is not an overwriting.
# p.join()

#  ----------- test share cuda tensor with queue ----------------

# import torch
# import torch.multiprocessing as mp
# import time


# def test_p(state_dict, idx):
#     for i in range(3):
#         print(idx, i, 'get tensor in process')
#         a = state_dict.get()
#         time.sleep(5)
#         print(idx, i, 'clone tensor in process')
#         b = a.clone()
#         time.sleep(5)
#         print(idx, i, 'put tensor in process')
#         state_dict.put(b)
#         time.sleep(5)
#         print(idx, i, 'del a tensor in process')
#         del a
#     time.sleep(500)
#     # time.sleep(10)
#     # print('del tensor in process')
#     # del a


# if __name__ == '__main__':
#     ctx = mp.get_context("spawn")
#     state_dict = ctx.Queue()
#     print('put tensor outside process')
#     state_dict.put(torch.rand([5000,5000]).cuda())
#     time.sleep(5)

#     p1 = ctx.Process(target=test_p, args=(state_dict,1))
#     p1.start()
#     # time.sleep(40)
#     # print('-'*40)
#     p2 = ctx.Process(target=test_p, args=(state_dict,2))
#     p2.start()

#     p1.join()
#     p2.join()


# ------------- test put multiple ----------------
# import torch
# import torch.multiprocessing as mp
# import time


# def test_p(state_dict, idx):
#     while not state_dict.empty():
#         print(state_dict.get())


# if __name__ == '__main__':
#     ctx = mp.get_context("spawn")
#     state_dict = ctx.Queue()

#     [state_dict.put(i) for i in [1,2,3]]
#     p1 = ctx.Process(target=test_p, args=(state_dict,1))
#     p1.start()
#     p1.join()

# ------------ test share cuda tensor -----------------
import torch
import torch.multiprocessing as mp
import time



def test_p(state_dict):
    for i in range(3):
        print(state_dict['t'])
        time.sleep(10)

def state_dict_inplace_update(old_state_dict, new_state_dict):
    for key in old_state_dict.keys():
        old_state_dict[key].data.copy_(new_state_dict[key].data.to(old_state_dict[key].device))

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    t = torch.tensor([1,2,3])
    
    t = t.to(torch.device('cuda'))
    t = t.share_memory_()

    state_dict = {'t': t}

    p = mp.Process(target=test_p, args=(state_dict,))
    p.start()

    time.sleep(5)
    state_dict_inplace_update(state_dict, {'t':torch.tensor([4,5,6])})
    # you can NOT do: t = torch.tensor([4,5,6]) because it is not an overwriting.

    time.sleep(15)
    state_dict_inplace_update(state_dict, {'t':torch.tensor([7,8,9])})

    p.join()

