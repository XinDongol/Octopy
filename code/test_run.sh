python main.py \
--model TestNet \
--dataset test \
--iid \
--num_users 10 \
--user_fraction 0.5 \
--lr 0.1 \
--local_epoch 3 \
--local_batchsize 1 \
--num_gpu 2 \
--num_local_models_per_gpu 2 \
--num_rounds 100 \
--data_dir /n/holyscratch01/kung_lab/xin/cifar_data \
--expr_dir /n/holyscratch01/kung_lab/xin/octopy_results/testrun

