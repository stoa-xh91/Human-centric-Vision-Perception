_base_ = [
    '../_base_/datasets/fashionpedia.py',
    '../_base_/models/ke_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_fashion_1x.py',
    '../_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadFourierFashionImageFromFile',
         aug_type=['sketch', 'anime']),
    dict(type='LoadFashionAnnotations',
         with_bbox=True,
         with_mask=True,
         with_attribute=True,
         with_human=True),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1,1)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img',
                               'aug_img',
                               'gt_bboxes', 
                               'gt_labels',
                               'gt_masks',
                               'gt_attributes']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline))

model = dict(
    type='CausalKERCNN',
    use_aug=True,
    roi_head=dict(
        type='CausalFashionRoIHead',
        use_aug=True))

custom_hooks = []
evaluation = dict(interval=100, metric='bbox')
checkpoint_config = dict(interval=2)