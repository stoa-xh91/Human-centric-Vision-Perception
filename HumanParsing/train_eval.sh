# DPAL Tiny
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=13920 --nproc_per_node 4 train.py --arch dpal_vit_tiny --imagenet-pretrain /path-to/DPAL-ViT-T.pth --batch-size 24 --learning-rate 7e-4 --weight-decay 0 --syncbn --lr_divider 500 --cyclelr_divider 2 --warmup_epochs 30 --epochs 150 --schp-start 120 --input-size 576,384 --log-dir ./logs/dpal-vit-t --data-dir ./data/LIP/LIP --optimizer adamw

CUDA_VISIBLE_DEVICES=0 python evaluate.py --arch dpal_vit_tiny --data-dir data/LIP/LIP --model-restore ./logs/dpal-vit-t/schp_4_checkpoint.pth.tar --input-size 576,384 --multi-scales 0.5,0.75,1.0,1.25,1.5 --flip

# DPAL Small
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=13924 --nproc_per_node 4 train.py --arch dpal_vit_small --imagenet-pretrain /path-to/DPAL-ViT-S.pth --batch-size 24 --learning-rate 7e-4 --weight-decay 0 --syncbn --lr_divider 500 --cyclelr_divider 2 --warmup_epochs 30 --epochs 150 --schp-start 120 --input-size 576,384 --log-dir ./logs/dpal-vit-s --data-dir ./data/LIP/LIP --optimizer adamw

# DPAL Base
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=13924 --nproc_per_node 4 train.py --arch dpal_vit_base --imagenet-pretrain /path-to/DPAL-ViT-B.pth --batch-size 24 --learning-rate 7e-4 --weight-decay 0 --syncbn --lr_divider 500 --cyclelr_divider 2 --warmup_epochs 30 --epochs 150 --schp-start 120 --input-size 576,384 --log-dir ./logs/dpal-vit-b --data-dir ./data/LIP/LIP --optimizer adamw

# Swin Tiny
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port=13920 --nproc_per_node 8 train.py --arch swin_small --imagenet-pretrain /path-to/DPAL-Swin-T.pth --batch-size 24 --learning-rate 7e-4 --weight-decay 0 --syncbn --lr_divider 500 --cyclelr_divider 2 --warmup_epochs 30 --epochs 120 --schp-start 90 --input-size 576,384 --log-dir ./logs/saipv2/dpal-swin-t --data-dir ./data/LIP/LIP --optimizer adamw

# Swin Small
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port=13920 --nproc_per_node 8 train.py --arch swin_small --imagenet-pretrain /path-to/DPAL-Swin-S.pth --batch-size 16 --learning-rate 7e-4 --weight-decay 0 --syncbn --lr_divider 500 --cyclelr_divider 2 --warmup_epochs 30 --epochs 120 --schp-start 90 --input-size 576,384 --log-dir ./logs/saipv2/dpal-swin-s --data-dir ./data/LIP/LIP --optimizer adamw


