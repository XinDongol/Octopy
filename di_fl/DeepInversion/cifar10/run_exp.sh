# python boostrap.py \
#     --resume 0 \
#     --batch_size 256 \
#     --epoch 10 \
#     --lr 0.01 \
#     --subset_pct 0.1 \
#     --di_batch_size 1024 \
#     --num_di_batch 1 \
#     --di_milestone '1,2' \
#     --logdir ./test
     
    


# python non_iid_di.py \
#     --logdir ./test \
#     --rounds 100 \
#     --num_devices 2 \
#     --device_pct 1 \
#     --non_iid 1 \
#     --local_epochs 1 \
#     --local_lr 0.01 \
#     --local_bsz 64 \
#     --local_di 0 \
#     --central_di -1 \
#     --central_di_batch_size 300