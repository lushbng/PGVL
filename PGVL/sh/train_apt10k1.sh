cd /home/lsb/CLAMP-main1
pip install -v -e .
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTL_ap10k_256x256.py 2 "0,1"

#bash tools/dist_test.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py   work_dirs/CLAMP_ViTB_ap10k_256x256/best_AP_epoch_*.pth 2 "2,3"
