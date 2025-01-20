cd /home/ubuntu/下载/Network_Based_PGVL
cd ./mmcv
pip install -v -e .
cd ../
pip install -v -e .
bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/PGVL_ViTB_coco_256X192.py 1 "0"



