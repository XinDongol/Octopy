# python boostrap.py \
#     --batch_size 256 \
#     --epoch 0 \
#     --lr 0.01 \
#     --subset_pct 0.1 \
#     --di_batch_size 256 \
#     --num_di_batch 200 \
#     --logdir ./pct_iid_01 \
#     --resume 1


CUDA_VISIBLE_DEVICES=7 python non_iid_di.py \
    --logdir ./non_iid_di_exp/10d-5ep-ld1-cd1 \
    --rounds 100 \
    --num_devices 10 \
    --device_pct 1 \
    --non_iid 1 \
    --local_epochs 5 \
    --local_lr 0.01 \
    --local_bsz 64 \
    --local_di 1 \
    --central_di 1 \
    --di_batch_size 100 &\