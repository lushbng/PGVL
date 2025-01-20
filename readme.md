# Parse Graph-Based Visual-Language Fusion for Human Pose Estimation (PGVL)



![Illustrating the architecture of the proposed RMPG](figs/feature_parse.jpg)
## Main Results
### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset
| Method            | Input size | Backbone |    AP |config|log|weight|
|--------------------|------------|--------|-------|----|----|-------|
| Ours    |    256x192   |   ViT-B | 0.747  |[config](RMPG_mmpose_based/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py)|[log](https://huggingface.co/shhibbnglulul/PGVL/blob/main/20241220_225929_coco_ViT_B.log)|[weight](https://huggingface.co/shhibbnglulul/PGVL/blob/main/best_AP_epoch_210_coco_ViT_B.pth)
| Ours    |    256x192   |   ViT-L | 0.770  |[config](RMPG_mmpose_based/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py)|[log](https://huggingface.co/shhibbnglulul/PGVL/blob/main/20241224_154254_COCO_VIT_L.log)|[weight](https://huggingface.co/shhibbnglulul/PGVL/blob/main/best_AP_epoch_190_COCO_VIT_L.pth)

### Results on CrowdPose test dataset
| Method            | Input size | Backbone |    AP |config|log|weight|
|--------------------|------------|--------|-------|----|----|-------|
| Ours    |    256x192   |   ViT-B | 0.678  |[config](RMPG_mmpose_based/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py)|[log](https://huggingface.co/shhibbnglulul/PGVL/blob/main/20241219_211951_CP_VIT_B.log)|[weight](https://huggingface.co/shhibbnglulul/PGVL/blob/main/best_AP_epoch_200__CP_VIT_B.pth)

### Results on MPII val dataset without multi-scale testing
| Method            | Input size | Backbone |    PCKh@0.5 |config|log|weight|
|--------------------|------------|--------|-------|----|----|-------|
| Ours    |    256x256   |   ViT-B | 0.914  |[config](RMPG_mmpose_based/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py)|[log](https://huggingface.co/shhibbnglulul/PGVL/blob/main/20250111_141721_MPII_VIT_B.log)|[weight](https://huggingface.co/shhibbnglulul/PGVL/blob/main/best_PCKh_epoch_200_MPII_VIT_B.pth)

### Results on AP-10K val dataset
| Method            | Input size | Backbone |    AP |config|log|weight|
|--------------------|------------|--------|-------|----|----|-------|
| Ours    |    256x256   |   ViT-B | 0.780  |[config](RMPG_mmpose_based/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py)|[log](https://drive.google.com/file/d/1MB_xKSj3cfeqqgqNnBThH-TUw0Lj3B5n/view?usp=drive_link)|[weight](https://drive.google.com/file/d/1EE320Ea5a9Yi3Ywpur1KPgElGSZ0vPnJ/view?usp=drive_link)
| Ours    |    256x256   |   ViT-L | 0.822  |[config](RMPG_mmpose_based/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py)|[log](https://drive.google.com/file/d/1MB_xKSj3cfeqqgqNnBThH-TUw0Lj3B5n/view?usp=drive_link)|[weight](https://drive.google.com/file/d/1EE320Ea5a9Yi3Ywpur1KPgElGSZ0vPnJ/view?usp=drive_link)

### Results on AnimalPose val dataset
| Method            | Input size | Backbone |    AP |config|log|weight|
|--------------------|------------|--------|-------|----|----|-------|
| Ours    |    256x256   |   ViT-B | 0.790  |[config](RMPG_mmpose_based/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py)|[log](https://drive.google.com/file/d/1MB_xKSj3cfeqqgqNnBThH-TUw0Lj3B5n/view?usp=drive_link)|[weight](https://drive.google.com/file/d/1EE320Ea5a9Yi3Ywpur1KPgElGSZ0vPnJ/view?usp=drive_link)



## Quick start
### For installation and environment setup, Please refer to mmpose (https://github.com/open-mmlab/mmpose) and vitpose (https://github.com/ViTAE-Transformer/ViTPose)
### For the numbers and depth of parse graph, please modify gp, ew and gp_list in the corresponding configuration file
### Training and Testing
```
cd RMPG_mmpose_based/sh
nohup ./sh/train_coco_50.sh>out.log 2>&1 &
nohup ./sh/train_coco_101.sh>out.log 2>&1 &
nohup ./sh/train_hourglass.sh>out.log 2>&1 &
nohup ./sh/test.sh>out.log 2>&1 &

cd RMPG_vitpose_based/sh
nohup ./sh/train.sh>out.log 2>&1 &
nohup ./sh/test.sh>out.log 2>&1 &
```


### Citation
If you use our code or models in your research, please cite with:
```
@article{PGBS,
	title={Human Pose Estimation via Parse Graph of Body Structure},
	author={Liu, Shibang and Xie, Xuemei and Shi, Guangming},
	journal=TCSVT,
	year={2024},
	publisher={IEEE}
}

``` 
