cd /home/lsb/CLAMP-main1
cd mmcv
pip  install -v -e .
cd ..
pip  install -v -e .
bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/crowdpose/CLAMP_ViTB_crowdpose_256X192.py 2 "4,5"

cd /home/lsb/CLAMP-main2
cd mmcv
pip  install -v -e .
cd ..
pip  install -v -e .
bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/crowdpose/CLAMP_ViTB_crowdpose_256X192.py 2 "4,5"

# bash tools/dist_test.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/crowdpose/CLAMP_ViTB_crowdpose_256X192.py work_dirs/CLAMP_ViTB_crowdpose_256X192/best_AP_epoch_*.pth 2 "2,3"