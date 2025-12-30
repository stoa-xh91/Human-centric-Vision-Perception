# pose
TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_vit_tiny_8xb64-210e_coco-256x192_wdiffusion.py 4

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_vit_tiny_8xb64-210e_coco-256x192.py 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29500 bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_vit_small_8xb64-210e_coco-256x192.py 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_vit_base_8xb64-210e_coco-256x192.py 4

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29500 bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_swin_tiny_8xb64-210e_coco-256x192.py 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_swin_small_8xb64-210e_coco-256x192.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_swin_base_8xb64-210e_coco-256x192.py 4
