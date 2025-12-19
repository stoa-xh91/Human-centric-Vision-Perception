_base_ = [
    '../_base_/datasets/fashionpedia.py',
    '../_base_/models/ke_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_fashion_1x.py',
    '../_base_/default_runtime.py'
]
pretrained = 'models/imagenet1k/edgenext_small.pth'
model = dict(
    backbone=dict(
        type='EdgeNeXt',
        depths=[3, 3, 9, 3], 
        dims=[48, 96, 160, 304], 
        expan_ratio=4,
        global_block=[0, 1, 1, 1],
        global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
        use_pos_embd_xca=[False, True, False, False],
        kernel_sizes=[3, 5, 7, 9],
        d2_scales=[2, 2, 3, 4],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(
        in_channels=[48, 96, 160, 304])
)
custom_hooks = []
evaluation = dict(interval=100, metric='bbox')
checkpoint_config = dict(interval=2)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0005,
    betas=(0.9, 0.999),
    weight_decay=0.05
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadFashionAnnotations',
         with_bbox=True,
         with_mask=True,
         with_attribute=True,
         with_human=True),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 1.5), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1,1)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 
                               'gt_bboxes', 
                               'gt_labels',
                               'gt_attributes',
                               'gt_masks']),
]

data = dict(samples_per_gpu=2,
    workers_per_gpu=2,train=dict(pipeline=train_pipeline))


# fp16 = dict(loss_scale=dict(init_scale=512))