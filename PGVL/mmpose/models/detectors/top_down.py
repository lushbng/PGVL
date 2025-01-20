# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .untils import tokenize

from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow
from mmcv.runner import load_checkpoint_cococlip

from mmpose.core import imshow_bboxes, imshow_keypoints
from .. import builder
from ..builder import POSENETS
from .base import BasePose
import logging
import pdb

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class TopDown(BasePose):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        super().__init__()
        self.fp16_enabled = False

        self.backbone = builder.build_backbone(backbone)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if keypoint_head is not None:
            keypoint_head['train_cfg'] = train_cfg
            keypoint_head['test_cfg'] = test_cfg

            if 'loss_keypoint' not in keypoint_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for TopDown is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint_head['loss_keypoint'] = loss_pose

            self.keypoint_head = builder.build_head(keypoint_head)

        self.init_weights(pretrained=pretrained)

    @property
    def with_neck(self):
        """Check if has neck."""
        return hasattr(self, 'neck')

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
        if self.with_neck:
            self.neck.init_weights()
        if self.with_keypoint:
            self.keypoint_head.init_weights()

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C (Default: 3)
            img height: imgH
            img width: imgW
            heatmaps height: H
            heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if `return loss` is true, then return losses.
                Otherwise, return predicted poses, boxes, image paths
                and heatmaps.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output = self.keypoint_head(output)

        # if return loss
        losses = dict()
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output, target, target_weight)
            losses.update(keypoint_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output, target, target_weight)
            losses.update(keypoint_accuracy)

        return losses

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output_heatmap = self.keypoint_head.inference_model(
                features, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)
            if self.with_neck:
                features_flipped = self.neck(features_flipped)
            if self.with_keypoint:
                output_flipped_heatmap = self.keypoint_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                output_heatmap = (output_heatmap +
                                  output_flipped_heatmap) * 0.5

        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, output_heatmap, img_size=[img_width, img_height])
            result.update(keypoint_result)

            if not return_heatmap:
                output_heatmap = None

            result['output_heatmap'] = output_heatmap

        return result

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Output heatmaps.
        """
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='TopDown')
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])

        if bbox_result:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            imshow_bboxes(
                img,
                bboxes,
                labels=bbox_labels,
                colors=bbox_color,
                text_color=text_color,
                thickness=bbox_thickness,
                font_scale=font_scale,
                show=False)

        if pose_result:
            imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                             pose_kpt_color, pose_link_color, radius,
                             thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img


@POSENETS.register_module()
class PGVL(BasePose):
    def __init__(self,
                 backbone,
                 text_encoder,
                 class_names,
                 context_length,
                 ALL_LINEAR=False,
                 score_concat_index=3,
                 identity_head=None,
                 upconv_head=None,
                 token_embed_dim=512,
                 text_dim=1024,
                 clip_pretrained=None,
                 matching_only=False,
                 visual_dim=256,
                 CL_ratio=1.0,
                 prompt_encoder=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 loss_pose=None,
                 pretrained=None):
        super().__init__()
        self.fp16_enabled = False
        self.backbone = builder.build_backbone(backbone)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if keypoint_head is not None:
            keypoint_head['train_cfg'] = train_cfg
            keypoint_head['test_cfg'] = test_cfg

            if 'loss_keypoint' not in keypoint_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for TopDown is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint_head['loss_keypoint'] = loss_pose

            self.keypoint_head = builder.build_head(keypoint_head)

        if text_encoder is not None:
            self.text_encoder = builder.build_backbone(text_encoder)



        self.with_prompt_encoder = False
        if prompt_encoder is not None:
            self.prompt_encoder = builder.build_backbone(prompt_encoder)
            self.with_prompt_encoder = True

        self.init_weights(pretrained=None, clip_pretrained=clip_pretrained)

        self.context_length = context_length
        self.score_concat_index = score_concat_index

        self.with_identity_head = False
        self.with_upconv_head = False
        self._init_identity_head(identity_head)
        # self._init_upconv_head(upconv_head)

        self.class_names = class_names
        self.matching_only = matching_only
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names]) #类别标签=语言
        self.num_classes = len(self.class_names)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.text_projection = nn.Parameter(torch.empty(text_encoder['embed_dim'], visual_dim))
        # nn.init.normal_(self.text_projection, std=text_encoder['embed_dim'] ** -0.5)
        self.CL_visual = nn.CrossEntropyLoss(reduce=False)
        self.CL_text = nn.CrossEntropyLoss(reduce=False)
        self.CL_ratio = CL_ratio

        context_length = self.text_encoder.context_length - self.context_length
        self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)

        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-3)

        self.my=EmbeddingProcessor(  # 训练的时候才初始化
            parse_dim_list=[768,512,1024], ew=[2,2,2], gp_list=[[2,2],[2,2],[2,2]], num_heads=2, qkv_bias=True, qk_scale=None,
            attn_drop=0, drop=0, attn_head_dim=None,num_blocks=1,target_dim=768,ALL_LINEAR=ALL_LINEAR)

        # self.my_down=nn.Linear(768,512)

        # self.down=nn.Conv2d(768,512,kernel_size=1)
        self.my_up=nn.Linear(512,768)

    @property
    def with_keypoint(self):
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None, clip_pretrained=None):
        self.backbone.init_weights(pretrained=clip_pretrained)
        if self.with_keypoint:
            self.keypoint_head.init_weights()
        self.text_encoder.init_weights()

    def load_checkpoint(self,
                        filename,
                        map_location='cpu',
                        strict=False,
                        revise_keys=[(r'^module.', '')]):
        logger = logging.getLogger()
        return load_checkpoint_cococlip(
            self,
            filename,
            map_location,
            strict,
            logger,
            revise_keys=revise_keys)

    def _init_identity_head(self, identity_head):
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = builder.build_head(identity_head)

    def _init_upconv_head(self, upconv_head):
        if upconv_head is not None:
            self.with_upconv_head = True
            self.upconv_head = builder.build_head(upconv_head)

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def spatial_adapt(self, x):
        visual_embeddings = x.clone()# bs 768 16 16
        # visual_embeddings = self.upv(visual_embeddings)#bs 128 16 16
        B, C, H, W = visual_embeddings.shape
        text_embeddings = self.text_encoder(self.texts.to(visual_embeddings.device), self.contexts).expand(B, -1, -1)#CLIPTextContextEncoder

        # model the relation of prompts
        if self.with_prompt_encoder:
            text_embeddings = self.prompt_encoder(text_embeddings)


        prompt_embeddings = text_embeddings# 


        upt=self.my_up(prompt_embeddings)
        my=self.my(upt,visual_embeddings.view(B,C,-1).permute(0,2,1)) # b 17 512; b 16*16 512

        t_res=my[0]#768 可考虑要不要加t_source


        v_res=my[1].permute(0,2,1).view(B,C,H,W)


        return t_res, v_res

    def feature_adapt(self, visual_embeddings, text_embeddings, target, target_weight):
        # (Batch, C, H, W) (Batch, 256, 64, 64) for x[0]
        B, C, H, W = visual_embeddings.shape
        # (Batch, K, D) (Batch, 17, 1024)
        B, K, D = text_embeddings.shape
        # (Batch, K, D) -> (Batch, K, C)
        if D != C:
            text_embeddings = text_embeddings @ self.text_projection

        target_mask = torch.where(target == 1, 1, 0)
        # (Batch, K, H, W, C) -> (Batch, K, C)
        visual_embeddings = torch.sum(torch.einsum('bkhw,bchw->bkhwc', target_mask, visual_embeddings), dim=(2, 3))

        visual_embeddings = F.normalize(visual_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * torch.einsum('bhc,bwc->bhw', visual_embeddings, text_embeddings)
        logits_per_text = logits_per_image.transpose(1, 2).contiguous()

        losses = dict()
        labels = torch.arange(K, device=logits_per_image.device).expand(B, -1)
        loss_visual = self.CL_visual(logits_per_image, labels) * target_weight.squeeze()
        loss_text = self.CL_text(logits_per_text, labels) * target_weight.squeeze()
        contrastive_loss = (loss_visual.mean() + loss_text.mean()) / 2

        losses['feature_loss'] = contrastive_loss * self.CL_ratio#

        return losses

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        target, target_down = target
        target_weight, target_down_weight = target_weight
        x_list = self.backbone(img)
        text_embeddings, output_vsf = self.spatial_adapt(x_list)#视觉文本融合，针对文本的交叉注意力

        if self.with_keypoint:
            output = self.keypoint_head(output_vsf)

        # if return loss
        losses = dict()
        contrastive_loss = self.feature_adapt(output_vsf, text_embeddings, target_down, target_down_weight)
        losses.update(contrastive_loss)#权重为 self.CL_ratio=0.0005
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output, target, target_weight)#weight=3
            losses.update(keypoint_losses)
            if not self.matching_only:
                keypoint_accuracy = self.keypoint_head.get_accuracy(
                    output, target, target_weight)#
                losses.update(keypoint_accuracy)

        return losses

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        img_metas=img_metas.data[0]
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features = self.backbone(img)

        text_embeddings, features = self.spatial_adapt(features)

        if self.with_keypoint:
            if not self.matching_only:
                output_heatmap = self.keypoint_head.inference_model(
                    features, flip_pairs=None)
            else:
                assert self.with_upconv_head
                output_heatmap = self.upconv_head.inference_model(
                    score_map, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)

            text_embeddings, features_flipped = self.spatial_adapt(features_flipped)
            if self.with_keypoint:
                if not self.matching_only:
                    output_flipped_heatmap = self.keypoint_head.inference_model(
                        features_flipped, img_metas[0]['flip_pairs'])
                else:
                    assert self.with_upconv_head
                    output_flipped_heatmap = self.upconv_head.inference_model(
                        score_map_flipped, img_metas[0]['flip_pairs'])
                output_heatmap = (output_heatmap +
                                  output_flipped_heatmap) * 0.5

        if self.with_keypoint:
            if not self.matching_only:
                keypoint_result = self.keypoint_head.decode(
                    img_metas, output_heatmap, img_size=[img_width, img_height])
            else:
                keypoint_result = self.upconv_head.decode(
                    img_metas, output_heatmap, img_size=[img_width, img_height])
            result.update(keypoint_result)

            if not return_heatmap:
                output_heatmap = None

            result['output_heatmap'] = output_heatmap

        return result

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='TopDown')
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])

        if bbox_result:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            imshow_bboxes(
                img,
                bboxes,
                labels=bbox_labels,
                colors=bbox_color,
                text_color=text_color,
                thickness=bbox_thickness,
                font_scale=font_scale,
                show=False)

        if pose_result:
            imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                             pose_kpt_color, pose_link_color, radius,
                             thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
class parse_my(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None, ew=2, gp_list=[2, 2]
                 ):
        super().__init__()
        self.primi_dim = dim // num_heads
        self.gp = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)


        self.self_attnt = Attention_my(
            dim, num_heads=num_heads, qkv_bias=True)
        self.self_attnv = Attention_my(
            dim, num_heads=num_heads, qkv_bias=True
        )
        self.ew = ew
        self.cross_attnt = Attention(dim, num_heads=2, proj_drop=0.1)
        self.cross_attnv = Attention(dim, num_heads=2, proj_drop=0.1)

        self.downv=nn.Linear(dim*2,dim)
        self.downt=nn.Linear(dim*2,dim)
        if ew <= 0:
            self.block = nn.ModuleList([nn.Identity() for i in range(num_heads)])

        else:
            self.block = nn.ModuleList(
                [parse_my(dim=self.primi_dim, num_heads=gp_list[ew - 1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                          qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=attn_drop, act_layer=act_layer,
                          norm_layer=norm_layer, attn_head_dim=attn_head_dim, ew=ew - 1,
                          gp_list=gp_list) for i in range(num_heads)])

    def forward(self, x):  # x=[x_t,x_v]
        pre_list_t = torch.chunk(x[0].clone(), self.gp, dim=2)
        out_list_t = []

        pre_list_v = torch.chunk(x[1].clone(), self.gp, dim=2)
        out_list_v = []
        for i in range(self.gp):
            t = pre_list_t[i]
            v = pre_list_v[i]
            # if self.ew==0:
            #     res_list=[t.clone(),v.clone()]
            # else:
            res_list = self.block[i]([t.clone(), v.clone()])
            out_list_t.append(res_list[0].clone())
            out_list_v.append(res_list[1].clone())
        ot = torch.cat(out_list_t, dim=2)
        ov = torch.cat(out_list_v, dim=2)

        qt = k = v = self.norm1(ot)
        ot_self = self.self_attnt(qt)  # 文本的节点的互注意力，

        qv = k = v = self.norm2(ov)
        ov_self = self.self_attnv(qv)  # 视觉节点的互注意力，

        ot_cross =  self.cross_attnt(qt, qv, qv) # 交叉注意力，对文本加强
        ov_cross =  self.cross_attnv(qv, qt, qt)  # 交叉注意力，对视觉加强

        t_res=torch.cat((ot_self,ot_cross),dim=-1)
        t_res=self.downt(t_res)+ot

        v_res=torch.cat((ov_self,ov_cross),dim=-1)
        v_res=self.downv(v_res)+ov
        return [t_res, v_res]
class Attention_my(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv = qkv.reshape(3, B, N * self.num_heads, C // self.num_heads)  # 3，B,heads*N，C//heads
        q, k, v = qkv[0], qkv[1], qkv[2]
        # qkv=x.clone().reshape(B,N*self.num_heads,C//self.num_heads)
        # q, k, v = qkv, qkv, qkv  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).reshape(B, N, self.num_heads, C // self.num_heads).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EmbeddingProcessor(nn.Module):
    def __init__(self, parse_dim_list=[768], ew=[2], gp_list=[[2, 2]],
                 num_heads=2, qkv_bias=True, qk_scale=None, attn_drop=0, drop=0,
                 attn_head_dim=None, target_dim=768, num_blocks=3,ALL_LINEAR=False):
        """
        :param parse_dim_list: 每个parse子层的输入维度列表
        :param num_blocks: parse层的数量
        :param target_dim: 拼接后降维到的目标维度
        """
        super(EmbeddingProcessor, self).__init__()

        self.parse_dim_list = parse_dim_list
        self.num_blocks = num_blocks

        # 生成每个parse层，根据parse_dim_list控制每个parse子层的输入维度
        self.parse_layers = nn.ModuleList([
            nn.ModuleList([  # 每个parse层包含多个parse子层
                parse_my(  # 在每一层都初始化一个不同的parse_my
                    dim=parse_dim_list[i], ew=ew[i], gp_list=gp_list[i], num_heads=num_heads,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                    drop=drop, attn_head_dim=attn_head_dim
                ) for i in range(len(parse_dim_list))  # 根据parse_dim_list的长度初始化多个parse子层
            ]) for _ in range(num_blocks)  # num_blocks表示parse的层数
        ])

        # 降维层：根据拼接的维度设置目标维度
        self.linear_layers_t = nn.ModuleList([
            nn.Identity() if sum(parse_dim_list) == target_dim else nn.Linear(sum(parse_dim_list), target_dim)
            for _ in range(num_blocks)
        ])
        self.linear_layers_v = nn.ModuleList([
            nn.Identity() if sum(parse_dim_list) == target_dim else nn.Linear(sum(parse_dim_list), target_dim)
            for _ in range(num_blocks)
        ])
        # MLP层和LayerNorm层
        self.mlp_layers_t = nn.ModuleList([
            Mlp(in_features=target_dim, hidden_features=target_dim * 4, act_layer=nn.GELU, drop=0.1) for _ in
            range(num_blocks)
        ])
        self.norm_layers_t = nn.ModuleList([nn.LayerNorm(target_dim) for _ in range(num_blocks)])

        self.mlp_layers_v = nn.ModuleList([
            Mlp(in_features=target_dim, hidden_features=target_dim * 4, act_layer=nn.GELU, drop=0.1) for _ in
            range(num_blocks)
        ])
        self.norm_layers_v = nn.ModuleList([nn.LayerNorm(target_dim) for _ in range(num_blocks)])

        self.initial_t_dim = target_dim  # 假设t_my的初始维度是768，您可以根据需要调整
        self.initial_v_dim = target_dim  # 假设v_my的初始维度是768，您可以根据需要调整

        # 为t_my和v_my分别添加不同的线性变换层

        # 为t_my和v_my分别添加不同的线性变换层
        if ALL_LINEAR:
            self.linear_transform_t = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.initial_t_dim, p_dim)
                    for p_dim in parse_dim_list
                ]) for _ in range(num_blocks)  # 每个block的parse层都有不同的线性变换
            ])

            self.linear_transform_v = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.initial_v_dim, p_dim)
                    for p_dim in parse_dim_list
                ]) for _ in range(num_blocks)  # 每个block的parse层都有不同的线性变换
            ])
        else:
            self.linear_transform_t = nn.ModuleList([
                nn.ModuleList([
                    nn.Identity() if p_dim == self.initial_t_dim else nn.Linear(self.initial_t_dim, p_dim)
                    for p_dim in parse_dim_list
                ]) for _ in range(num_blocks)  # 每个block的parse层都有不同的线性变换
            ])

            self.linear_transform_v = nn.ModuleList([
                nn.ModuleList([
                    nn.Identity() if p_dim == self.initial_v_dim else nn.Linear(self.initial_v_dim, p_dim)
                    for p_dim in parse_dim_list
                ]) for _ in range(num_blocks)  # 每个block的parse层都有不同的线性变换
            ])


    def forward(self, prompt_embeddings, vision_embeddings):
        """
        多次重复计算parse，并更新t_my和v_my。
        """
        t_my = prompt_embeddings
        v_my = vision_embeddings

        # Step 2: 多次迭代更新parse和计算t_my/v_my
        for i in range(len(self.parse_layers)):
            # 保存每个parse子层的结果，用于拼接
            parse_results_t = []
            parse_results_v = []

            # 每个parse层可以有多个子层，依次处理
            for j, parse_sublayer in enumerate(self.parse_layers[i]):
                # 根据parse_dim_list动态调整维度
                t_my_transformed = self.linear_transform_t[i][j](t_my)  # 当前层和子层的t_my维度调整
                v_my_transformed = self.linear_transform_v[i][j](v_my)  # 当前层和子层的v_my维度调整
                my = parse_sublayer([t_my_transformed, v_my_transformed]) # 这里假设parse的输入是t_my_transformed和v_my_transformed，可以根据实际需求进行调整
                parse_results_t.append(my[0])  # 假设每个parse子层输出的是一个tuple，我们取第一个
                parse_results_v.append(my[1])  # 假设每个parse子层输出的是一个tuple，我们取第一个

            # 拼接所有子层的结果
            concatenated_result_t = torch.cat(parse_results_t, dim=-1)  # 在最后一个维度拼接
            concatenated_result_v = torch.cat(parse_results_v, dim=-1)  # 在最后一个维度拼接

            # 降维
           # 降维（如果需要）
            reduced_result_t = self.linear_layers_t[i](concatenated_result_t)#B,N,768
            reduced_result_v = self.linear_layers_v[i](concatenated_result_v)#B,N,768

            # 更新t_my和v_my
            t_my = reduced_result_t + t_my
            v_my = reduced_result_v + v_my

            # 应用MLP和LayerNorm
            t_my = t_my + self.mlp_layers_t[i](self.norm_layers_t[i](t_my))
            v_my = v_my + self.mlp_layers_v[i](self.norm_layers_v[i](v_my))

        return t_my, v_my
