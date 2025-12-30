from mmpose.registry import MODELS
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
# from mmengine.model import BaseModule
from .base_backbone import BaseBackbone
import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import random
from typing import Sequence
import collections.abc as container_abcs
from itertools import repeat
from .vit import VisionTransformer
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from .denoiser import Denoiser
from .samplers import euler_sampler, euler_maruyama_sampler

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class DiffusionModule(nn.Module):
    def __init__(
            self,
            out_dim=768,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            **kwargs
            ):
        super().__init__()
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        block_kwargs = {"fused_attn": True, "qk_norm": False}
        self.denoiser = Denoiser(depth=1, hidden_size=out_dim, num_heads=12, **block_kwargs)

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, teacher_feat, student_feat):
        assert teacher_feat.shape == student_feat.shape, "特征维度不匹配"
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((teacher_feat.shape[0], 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((teacher_feat.shape[0], 1 ,1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=teacher_feat.device, dtype=teacher_feat.dtype)
        
        noises = torch.randn_like(teacher_feat)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        noisy_teacher_feat = alpha_t * teacher_feat + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * teacher_feat + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output = self.denoiser(noisy_teacher_feat, time_input.flatten(), student_feat)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        return denoising_loss.mean()

@MODELS.register_module()
class VisionTransformerDiffusion(BaseBackbone):
    """ Vision Transformer """
    def __init__(self, init_cfg, fronzen=False, **args):
        super(VisionTransformerDiffusion, self).__init__(init_cfg)
        self.backbone = VisionTransformer(**args)
        self.align_module = nn.Sequential(
            nn.Linear(args['embed_dim'], args['hidden_dim']),
            nn.ReLU(),
            nn.Linear(args['hidden_dim'], args['bottleneck_dim']),
            nn.ReLU(),
            nn.Linear(args['bottleneck_dim'], args['out_dim']),
            nn.LayerNorm(args['out_dim'])
        )
        self.diffusion_module = DiffusionModule(**args)
        # if fronzen:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False
        #     for param in self.align_module.parameters():
        #         param.requires_grad = False
        #     for param in self.diffusion_module.parameters():
        #         param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            checkpoint_model = torch.load(
                self.init_cfg['checkpoint'], map_location='cpu')
            print('load from:',self.init_cfg['checkpoint'],)
            if 'model' in checkpoint_model:
                param_dict = checkpoint_model['model']
            elif 'state_dict' in checkpoint_model:
                param_dict = checkpoint_model['state_dict']
            elif 'student' in checkpoint_model: ### for dino
                param_dict = checkpoint_model["student"]
            else:
                param_dict = checkpoint_model
            param_dict = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in param_dict.items()}
            msg = self.load_state_dict(param_dict, strict=False)
            print('load vit:', msg)
            
        else:
            super(VisionTransformerDiffusion, self).init_weights()

    def forward(self, x):
        b_feats_heat_map = self.backbone(x)
        B,C,H,W = b_feats_heat_map[0].shape
        b_feats = b_feats_heat_map[0].reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        b_feats_align = self.align_module(b_feats)
        _,_,C = b_feats_align.shape
        z = torch.randn_like(b_feats_align).cuda()
        samples = euler_sampler(model=self.diffusion_module.denoiser,
                                    latents=z,
                                    y=b_feats_align,
                                    num_steps=100,
                                    heun=False,
                                    cfg_scale=1.0,
                                    guidance_low=0.0,
                                    guidance_high=1.0,
                                    path_type="linear").to(torch.float32)
        samplers_heat_map = samples.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return tuple([samplers_heat_map])