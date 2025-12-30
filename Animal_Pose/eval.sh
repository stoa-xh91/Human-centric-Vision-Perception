# r50
python tools/test.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/res50_ap10k_256x256.py models/res50_ap10k_256x256-35760eb8_20211029.zip
# r101
python tools/test.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/res101_ap10k_256x256.py models/res101_ap10k_256x256-9edfafb9_20211029.zip
# hrnet32
python tools/test.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/hrnet_w32_ap10k_256x256.py models/hrnet_w32_ap10k_256x256-18aac840_20211029.zip
# hrnet48
python tools/test.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/hrnet_w48_ap10k_256x256.py models/hrnet_w48_ap10k_256x256-d95ab412_20211029.zip


# dino
python tools/test.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/saip_ap10k_256x256.py work_dirs/dino_ap10k_256x256/epoch_210.pth
# mae
python tools/test.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/saip_ap10k_256x256.py work_dirs/mae_ap10k_256x256/epoch_210.pth
# dino_mae
python tools/test.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/saip_ap10k_256x256.py work_dirs/dino_mae_ap10k_256x256/epoch_210.pth
# hap
python tools/test.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/saip_ap10k_256x256.py work_dirs/hap_ap10k_256x256/epoch_210.pth
# soilder
python tools/test.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/saip_ap10k_256x256.py work_dirs/soilder_ap10k_256x256/epoch_210.pth
# saip
python tools/test.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/saip_ap10k_256x256.py work_dirs/saip_ap10k_256x256/epoch_210.pth

# dino
python tools/test.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/saip_ap10k_256x256.py work_dirs/dino_ap10k_256x256/epoch_210.pth