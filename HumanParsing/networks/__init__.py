from __future__ import absolute_import
from networks.swin_transformer import swin_base_patch4_window7_224 as swin_base
from networks.swin_transformer import swin_small_patch4_window7_224 as swin_small
from networks.swin_transformer import swin_tiny_patch4_window7_224 as swin_tiny
from networks.tinyvit import tinyvit_5m, tinyvit_21m
from networks.vision_transformer import vit_base, vit_small, vit_tiny, dpal_vit_tiny, dpal_vit_small, dpal_vit_base, mmvit_03B
from networks.cspnext import cspnext_5m
from networks.edgenext import edgenext_small_hp
from networks.starnet import starnet_s3_hp

from networks.AugmentCE2P import resnet101
from networks.AugmentCE2P import resnet50
__factory = {
    'resnet101': resnet101,
    'resnet50': resnet50,
    'cspnext_5m':cspnext_5m,
    'edgenext_small':edgenext_small_hp,
    'starnet_s3':starnet_s3_hp,
    'swin_base': swin_base,
    'swin_tiny': swin_tiny,
    'swin_small': swin_small,
    'tinyvit_5m': tinyvit_5m,
    'tinyvit_21m': tinyvit_21m,
    'vit_tiny': vit_tiny,
    'vit_small':vit_small,
    'vit_base':vit_base,
    'dpal_vit_tiny':dpal_vit_tiny,
    'dpal_vit_small':dpal_vit_small,
    'dpal_vit_base':dpal_vit_base,
    'mmvit_03B':mmvit_03B
}


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model arch: {}".format(name))
    return __factory[name](*args, **kwargs)
