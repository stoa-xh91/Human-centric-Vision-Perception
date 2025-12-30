echo "=====================================dino====================================="
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/dino_ap10k_256x256.py 8
echo "=====================================mae====================================="
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/mae_ap10k_256x256.py 8
echo "=====================================dino_mae====================================="
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/dino_mae_ap10k_256x256.py 8
echo "=====================================hap====================================="
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/hap_ap10k_256x256.py 8
echo "=====================================solider====================================="
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/soilder_ap10k_256x256.py 8
echo "=====================================saip====================================="
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/saip_ap10k_256x256.py 8


CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/vit_tiny_ap10k_256x128.py 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29504 bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/vit_tiny_ap10k_256x128.py 4

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29507 bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/vit_small_ap10k_256x128.py 4

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29503 bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/vit_base_ap10k_256x128.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/vit_base_ap10k_256x128.py 4

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/vit_large_ap10k_256x128.py 4

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29502 bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/swin_tiny_ap10k_256x128.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29505 bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/swin_small_ap10k_256x128.py 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29500 bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/swin_base_ap10k_256x128.py 4