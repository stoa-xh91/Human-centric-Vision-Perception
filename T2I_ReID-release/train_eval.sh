#!/bin/bash
# DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=4 \
python train.py \
--name iira \
--img_aug \
--batch_size 128 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60 \
--lr 0.0001 \
--output_dir 'logs/dpal_vit-t' 


DATASET_NAME="ICFG-PEDES"

# CUDA_VISIBLE_DEVICES=9 \
# python train.py \
# --name iira \
# --img_aug \
# --batch_size 128 \
# --MLM \
# --dataset_name $DATASET_NAME \
# --loss_names 'sdm+mlm+id' \
# --num_epoch 60 \
# --lr 0.0001 \
# --output_dir 'logs/dpal_vit-t' 
# --resume_ckpt_file 'logs/dpal_vit-t/best.pth'