_base_ = ['../../../../_base_/datasets/crowdpose.py']
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10,max_keep_ckpts=6)
evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(type='AdamW',
                 lr=5e-4,
                 weight_decay=0.0001,
                 paramwise_cfg=dict(custom_keys={'text_encoder': dict(lr_mult=0.0),
                                                 'backbone': dict(lr_mult=0.1),
                                                 'norm': dict(decay_mult=0.)})
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

channel_cfg = dict(
    num_output_channels=14,
    dataset_joints=14,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
    ])

# model settings
model = dict(
    type='PGVL',
    clip_pretrained='pretrained/ViT-B-16.pt',
    context_length=5,
    text_dim=512,
    score_concat_index=3,
    visual_dim=512,
    CL_ratio=0.0005,
    parse_dim_list=[512],
    ew=[2],#解析图深度
    gp_list=[[2,2]],
    num_heads=2,
    target_dim=512,
    src_to_dim=[768,512],
    mode=5,
    class_names=['left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist',
                 'right wrist', 'left hip', 'right hip',
                 'left knee', 'right knee', 'left ankle',
                 'right ankle', 'upper head', 'upper neck'],
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=13,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        pretrained='pretrained/ViT-B-16.pt',
        style='pytorch'),
    prompt_encoder=dict(
        type='PromptEncoderWithoutPositionemb',
        prompt_num=14,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=1,
        embed_dim=512,
        style='pytorch'),
    context_decoder=None,
    backbone=dict(
        type='MY_VIT_VisionTransformer',
        pretrained='pretrained/mae_pretrain_vit_base.pth',
        img_size=(192, 256),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
        style='pytorch'),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        num_deconv_layers=2,
        num_deconv_filters=(192, 256),
        num_deconv_kernels=(4, 4),
        in_channels=512, #768+17
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True, loss_weight=1.0)),

    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))
data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    crowd_matching=False,
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file='data/crowd_pose/json/'
    'det_for_crowd_test_0.1_0.5.json',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=6,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2,downtarget=True,same_dim=False,short_dim=12, downsize=16),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = f'data/crowd_pose'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type='TopDownCrowdPoseDataset',
        ann_file=f'{data_root}/json/mmpose_crowdpose_trainval.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownCrowdPoseDataset',
        ann_file=f'{data_root}/json/mmpose_crowdpose_test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCrowdPoseDataset',
        ann_file=f'{data_root}/json/mmpose_crowdpose_test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}))
