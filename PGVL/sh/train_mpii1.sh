cd /home/lsb/CLAMP-main1
cd mmcv
pip  install -v -e .
cd ..
pip  install -v -e .
bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/CLAMP_ViTB_mpii_256x256.py 2 "4,5"


