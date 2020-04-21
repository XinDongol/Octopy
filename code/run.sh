python main.py \
--model CifarNet \
--dataset cifar10 \
--iid \
--num_users 50 \
--user_fraction 0.2 \
--lr 0.01 \
--local_epoch 50 \
--momentum 0.9 \
--optimizer SGD \
--local_batchsize 32 \
--num_gpu 2 \
--num_local_models_per_gpu 4 \
--num_rounds 200 \
--data_dir /home/jovyan/harvard-heavy/xin/cifar10_data \
--expr_dir /home/jovyan/harvard-heavy/xin/octopy_results/var_reg_1e-8

