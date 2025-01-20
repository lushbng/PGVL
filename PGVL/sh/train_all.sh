
cd /home/lsb/CLAMP-main3
cd ./mmcv
pip install -v -e .
cd ../
pip install -v -e .
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTL_ap10k_256x256.py 4 "0,1,2,3"

cd /home/lsb/CLAMP-main2
cd ./mmcv
pip install -v -e .
cd ../
pip install -v -e .
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTL_ap10k_256x256.py 4 "0,1,2,3"



cd /home/lsb/CLAMP-main4
cd ./mmcv
pip install -v -e .
cd ../
pip install -v -e .
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py 2 "2,3"  &

cd /home/lsb/CLAMP-main5
cd ./mmcv
pip install -v -e .
cd ../
pip install -v -e .
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py 2 "0,1"  &

wait

cd /home/lsb/CLAMP-main6
cd ./mmcv
pip install -v -e .
cd ../
pip install -v -e .
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py 2 "0,1" &

cd /home/lsb/CLAMP-main1
cd ./mmcv
pip install -v -e .
cd ../
pip install -v -e .
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/CLAMP_ViTB_animalpose_256x256.py 1 "2"  &

wait 