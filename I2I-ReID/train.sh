# Market DPAL-ViT-T
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/market/dpal_vit_tiny.yml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH '/path-to/DPAL-ViT-T.pth' OUTPUT_DIR './log/market/dpal-vit-t' SOLVER.BASE_LR 0.0002 SOLVER.OPTIMIZER_NAME 'AdamW'

# MSMT DPAL-ViT-T
# CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/msmt17/dpal_vit_tiny.yml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH '/path-to/DPAL-ViT-T.pth' OUTPUT_DIR './log/msmt17/dpal-vit-t' SOLVER.BASE_LR 0.0002 SOLVER.OPTIMIZER_NAME 'AdamW'

# Market DPAL-Swin-T
# CUDA_VISIBLE_DEVICES=6 python train.py --config_file configs/market/swin_tiny.yml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH '/path-to/DPAL-Swin-T.pth' OUTPUT_DIR './log/market/dpal-swin-t' SOLVER.BASE_LR 0.0002 SOLVER.OPTIMIZER_NAME 'AdamW'

# MSMT DPAL-Swin-T
# CUDA_VISIBLE_DEVICES=7 python train.py --config_file configs/msmt17/swin_tiny.yml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH '/path-to/DPAL-Swin-T.pth' OUTPUT_DIR './log/msmt17/dpal-swin-t' SOLVER.BASE_LR 0.0002 SOLVER.OPTIMIZER_NAME 'AdamW'
