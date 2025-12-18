_base_ = ['./crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman.py']

model = dict(
    backbone=dict(
        depth=18, init_cfg=dict(checkpoint='torchvision://resnet18')),
    neck=dict(
        in_channels=[64, 128, 256, 512]),
    )

