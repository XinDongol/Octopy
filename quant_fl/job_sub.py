#=================== =  = JOB LAUNCHER#Hongxu Yin, Jun 1, 2020
#This code generates the starting point for model inversion

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import subprocess
import sys, os
import time
import itertools
import numpy as np
import string

SUBMIT  = 'none'
#SUBMIT = 'tar'

if   SUBMIT         == '8211':
    data           = '/imagenet/train'
    cmd_forjob_pre = ''
    dataset        = '8211:/imagenet'
elif SUBMIT         == 'tar':
    data           = '/tmp/imagenet/ImageNet2012/ImageNet2012/'
    cmd_forjob_pre = 'mkdir /tmp/imagenet/; tar -C /tmp/imagenet -xvf /imagenet/imagenet2012.tar.gz; '
    dataset        = '34479:/imagenet'
elif SUBMIT         == 'none':
    cmd_forjob_pre = 'bash /myws/codes/ngc_comm/install.sh'
    dataset        = '34479:/imagenet'
else: 
	raise ValueError('SUMIT option not supported')

# Cluster configs
image     = "nvidian/pytorch:20.05-py3-base"
result    = "/result"
workspace = "ogosJhGySqGnwt1RllPkrg:/myws:RW"
# instance  = "ngcv1"
instance  = "dgx1v.16g.1.norm"



cmd_main = "cd /myws/codes/Octopy/quant_fl; python debias_bn_realfl.py \
--root ./v01_fl/{exp_name} \
--rounds {rounds} --num_devices {num_devices} --device_pct {device_pct} --non_iid {non_iid} \
--local_epoch {local_epoch} --local_lr {local_lr} --local_bsz {local_bsz} \
--use_di {use_di} --iterations_per_layer {iterations_per_layer} --inv_bsz {inv_bsz}"


if 1: 
	all_rounds      = [100]
	all_num_devices = [20, 100]
	all_device_pct  = [0.2, 1.]
	all_non_iid     = [0, 1]


	all_local_epoch = [1 , 5, 20]
	all_local_lr    = [0.01]
	all_local_bsz   = [32, 64]


	all_use_di               = [0, 1]
	all_iterations_per_layer = [2000]
	all_inv_bsz              = [2, 20]

SUBMIT_job  = True
all_configs = list(itertools.product(all_rounds, all_num_devices, all_device_pct, all_non_iid,\
				 all_local_epoch, all_local_lr, all_local_bsz, \
				 all_use_di, all_iterations_per_layer, all_inv_bsz))


num_configs = len(all_configs)
print('==> total number of configs: %d' % num_configs)
random_names = np.random.choice([''.join(i) for i in itertools.product(string.ascii_lowercase, repeat=3)], num_configs)

for idx, (rounds, num_devices, device_pct, non_iid, \
	local_epoch, local_lr, local_bsz, \
	use_di, iterations_per_layer, inv_bsz) in enumerate(all_configs):

	name = random_names[idx]

	command = cmd_main.format(exp_name = 'noniid{}/'.format(non_iid) + name,
								rounds               = rounds,
								num_devices          = num_devices,
								device_pct           = device_pct,
								non_iid              = non_iid,
								local_epoch          = local_epoch,
								local_lr             = local_lr,
								local_bsz            = local_bsz,
								use_di               = use_di,
								iterations_per_layer = iterations_per_layer,
								inv_bsz              = inv_bsz)

	full_cmd =  (   'ngc batch run --instance {instance} '\
										'--name {name} '\
										'--image {image} '\
										'--result {result} '\
										'--datasetid {dataset} '\
										'--workspace {workspace} '\
										'--team iva '\
										'--org nvidian '\
										'--commandline \'{final_command}\'').format(
										instance      = instance,
										name          = name,
										image         = image,
										result        = result,
										dataset       = dataset,
										workspace     = workspace,
										final_command = cmd_forjob_pre+'; '+command)

	print('\n\n==> Now Submitting job: ', full_cmd)
	# break
	print("\n\n")
	if SUBMIT_job: os.system(full_cmd)