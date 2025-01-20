# Refinement Module based on Parse Graph of Feature Map for Human Pose Estimation (RMPG)



![Illustrating the architecture of the proposed RMPG](figs/feature_parse.jpg)
## Main Results
### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset
| Method            | Input size | #Params | Backbone |    AP |config|log|weight|
|--------------------|------------|---------|--------|-------|----|----|-------|
| SimpleBaselines+RMPG<sub>gp=[2,2]     |    256x192 | 38.0M   |   ResNet-50 | 0.725  |[config](RMPG_mmpose_based/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py)|[log](https://drive.google.com/file/d/1MB_xKSj3cfeqqgqNnBThH-TUw0Lj3B5n/view?usp=drive_link)|[weight](https://drive.google.com/file/d/1EE320Ea5a9Yi3Ywpur1KPgElGSZ0vPnJ/view?usp=drive_link)
| SimpleBaselines+RMPG<sub>gp=[2,2]     |    256x192 | 57.0M   |   ResNet-101 | 0.731  |[config](RMPG_mmpose_based/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest101_8xb64-210e_coco-256x192.py)|[log](https://drive.google.com/file/d/1SA_t5bFIMwwbexxlOl_KtTSXqiJtb4yW/view?usp=drive_link)|[weight](https://drive.google.com/file/d/1KtBKTQaQZYwh7zjEl8ZvBOlC6OI0fwir/view?usp=drive_link)
| Hourglass+RMPG<sub>gp=[2,2]          |    256x256 | 98.0M   |   Hourglass52 | 0.740  |[config](RMPG_mmpose_based/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hourglass52_8xb32-210e_coco-256x256.py)|[log](https://drive.google.com/file/d/12rsx3LSPalwE-DUFQUOXDO4UyskZeclg/view?usp=drive_link)|[weight](https://drive.google.com/file/d/1SfRBxBQemHpi8rdD_QTnWWLxjiCj4Ll8/view?usp=drive_link)
| ViTPose+RMPG<sub>gp=[2,2]             |    256x192 | 117.9M   |   ViTPose-B| 0.761 |[config](RMPG_vitpose_based/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py)|[log](https://drive.google.com/file/d/13fMtpqVEYeoioFFzMbbOdvNaHPW6LFpd/view?usp=drive_link)|[weight](https://drive.google.com/file/d/1VCoZ2ftyxJrnI2310OGkssWnFuZJ8lSq/view?usp=drive_link)



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
