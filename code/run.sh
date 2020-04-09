python main.py \
--model CifarNet \
--dataset cifar10 \
--iid \
--num_users 10 \
--user_fraction 0.5 \
--lr 0.01 \
--local_epoch 10 \
--momentum 0.9 \
--optimizer SGD \
--local_batchsize 32 \
--num_gpu 2 \
--num_local_models_per_gpu 2 \
--num_rounds 200 \
--data_dir /n/holyscratch01/kung_lab/xin/cifar_data \
--expr_dir /n/holyscratch01/kung_lab/xin/octopy_results/test

