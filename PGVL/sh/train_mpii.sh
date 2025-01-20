cd /home/ubuntu/下载/Network_Based_PGVL
cd mmcv
pip  install -v -e .
cd ..
pip  install -v -e .
bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/PGVL_ViTB_mpii_256x256.py 1 "0"
