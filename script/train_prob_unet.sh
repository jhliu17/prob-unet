export CUDA_VISIBLE_DEVICES=$1

python run.py --mode train \
              --git_info `git describe --always`\
              --seed 2023 \
              --train_dataset_path /srv/disk00/junhal11/oct_understanding/data/2015_boe_chiu/2015_BOE_Chiu/train_set_wo_xobject.npz \
              --test_dataset_path /srv/disk00/junhal11/oct_understanding/data/2015_boe_chiu/2015_BOE_Chiu/test_set_wo_xobject.npz  \
              --batch_size 8 \
              --lr 1e-2 \
              --latent_size 16 \
              --epoch 300 \
              --number_worker 2 \
              --log_freq 10 \
              --eval_freq 50 \
              --save_ratio 0 \
              --ckpt_dir output/segmentation/prob_unet
