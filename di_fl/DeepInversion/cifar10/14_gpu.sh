CUDA_VISIBLE_DEVICES=6 python non_iid_di.py --logdir /n/holyscratch01/kung_lab/xin/fedbn_results/test_central_di/raspberry-mamba_2020.12.29_22.44 --rounds 100 --num_devices 20 --device_pct 0.5 --non_iid 1 --scheduler cosine --local_epochs 5 --local_lr 0.01 --local_bsz 64 --local_reset_optim 0 --di_lr 0.05 --di_steps 2000 --di_scheduler 0 --local_di 0 --local_bn_stat_epochs 0 --local_di_reset_optim 1 --local_di_batch_size 16 --local_di_celoss 1.0 --local_di_bnloss 10.0 --central_bn_update_celoss 1.0 --central_bn_update_lr 1e-3 --central_bn_update_momentum 0.9 --central_bn_update_epochs 80 --central_bn_update_scheduler 1 --central_di 4 --central_di_batch_size 256 --local_mix_bsz 64 --central_di_celoss 0.0 --central_di_bnloss 10.0 --central_di_reset_optim 1 --central_di_distill_temp 5.0 --central_di_distill_loss_scale 10.0 --print_local_test 1 --fedprox 0 --fedprox_mu 0 --reset_momentum 0 
