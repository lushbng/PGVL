cd /home/lsb/CLAMP-main1
cd ./mmcv
pip install -v -e .
cd ../
pip install -v -e .
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/CLAMP_ViTB_animalpose_256x256.py 2 "0,1"


# cd /home/lsb/CLAMP-main2
# cd ./mmcv
# pip install -v -e .
# cd ../
# pip install -v -e .
# bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py 2 "0,1" 

# cd /home/lsb/CLAMP-main2
# cd ./mmcv
# pip install -v -e .
# cd ../
# pip install -v -e .
# bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/CLAMP_ViTB_coco_256x192.py 4 "0,1,2,3" 

# cd /home/lsb/CLAMP-main3
# pip install -v -e .
# bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py 2 "0,1"

# cd /home/lsb/CLAMP-main4
# pip install -v -e .
# bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py 2 "0,1"
# bash tools/dist_test.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py   res/apt10k/train20.1/best_AP_epoch_200.pth 2 "0,1"

# cd /home/lsb/CLAMP-main2
# pip install -v -e .
# bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py 2 "2,3"

# bash tools/dist_test.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py   work_dirs/CLAMP_ViTB_ap10k_256x256/best_AP_epoch_*.pth 2 "2,3"

# cd /home/lsb/CLAMP-main3
# pip install -v -e .
# bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py 2 "2,3"

# bash tools/dist_test.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py   work_dirs/CLAMP_ViTB_ap10k_256x256/best_AP_epoch_*.pth 2 "2,3"

# cd /home/lsb/CLAMP-main4
# pip install -v -e .
# bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py 2 "2,3"

# bash tools/dist_test.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py   work_dirs/CLAMP_ViTB_ap10k_256x256/best_AP_epoch_*.pth 2 "2,3"