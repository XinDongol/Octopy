import GPUtil
import itertools
import string
import numpy as np
import time
import os

execute=True

if 1:
    all_lr = [0.001, 0.01]
    all_epochs = [100]
    all_baseline = [0, 1]
    all_T = [5., 10., 20.]
    all_lamb = [0.5, 1.0]
    all_kd_scale = [1.]

    
all_cmd = []

all_configs = list(itertools.product(all_lr, all_epochs, all_baseline, all_T, all_lamb, all_kd_scale))

print('===> total number of configs: %d'%len(all_configs))

cmd_main = 'python train_on_inverted.py --root ./distill/{exp_name} --lr {lr} --epochs {epochs} --baseline {baseline} --T {T} --lamb {lamb} --kd_scale {kd_scale}'

random_names = np.random.choice([''.join(i) for i in itertools.product(string.ascii_lowercase, repeat=3)], len(all_configs))

for idx, (lr, epochs, baseline, T, lamb, kd_scale) in enumerate(all_configs):
    name = random_names[idx]
    command = cmd_main.format(exp_name=name,
                              lr=lr,
                              epochs=epochs,
                              baseline=baseline,
                              T=T,
                              lamb=lamb,
                              kd_scale=kd_scale
                              )
    
    while 1:
        deviceIDs = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.3, maxMemory=0.3, includeNan=False, excludeID=[], excludeUUID=[])
        if len(deviceIDs)>0:
            pre_cmd = 'CUDA_VISIBLE_DEVICES=%s'%deviceIDs[0]
            final_cmd = pre_cmd + ' ' + command + ' &'
            print('\n\n==> Now Submitting Job: ', final_cmd)
            
            if execute:
                os.system(final_cmd)
            time.sleep(60)
            break
        
    time.sleep(60)
    
    
    
    

    
    
