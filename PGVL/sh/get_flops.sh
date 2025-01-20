
python tools/analysis/get_flops.py   configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/CLAMP_ViTB_mpii_256x256.py  --shape 256 256

# bash tools/dist_test.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py public-models/ViT-B/epoch_210.pth 4 "0,1,2,3"