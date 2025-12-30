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
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads,batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, key, value, return_attention=False):
        query = self.norm1(x)
        y, attn = self.attn(query, key, value)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        # 
        y = self.drop_path(y)
        x = x + y
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        h_out = (img_size[0] + 2 * 2 - 1 *
                     (patch_size - 1) - 1) // patch_size + 1
        w_out = (img_size[1] + 2 * 2 - 1 *
                    (patch_size - 1) - 1) // patch_size + 1
        self.init_out_size = (h_out, w_out)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,padding=2)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, out_size
    
def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    # The cubic interpolate algorithm only accepts float32
    dst_weight = torch.nn.functional.interpolate(
        src_weight.float(), size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    dst_weight = dst_weight.to(src_weight.dtype)

    return torch.cat((extra_tokens, dst_weight), dim=1)

@MODELS.register_module()
class VisionTransformer(BaseBackbone):
    """ Vision Transformer """
    def __init__(self, img_size=(224,224), patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, num_heads_in_last_block=12,mlp_ratio=4., out_indices=[3, 7, 11], out_type='cls_token', out_scales=[2, 1, 0.5], 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., with_cls_token=True,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, frozen_stages=-1, init_cfg=[
                    dict(type='Normal', std=0.001, layer=['Conv2d']),
                    dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
                 ], **kwargs):
        super(VisionTransformer, self).__init__(init_cfg)
        self.num_features = self.embed_dim = embed_dim
        self.num_extra_tokens = 1
        self.img_size = to_2tuple(img_size)
        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_resolution = self.patch_embed.init_out_size
        self.with_cls_token = with_cls_token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        else:
            self.cls_token = None
            self.num_extra_tokens = 0
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_extra_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.num_layers = depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth-1)]+[ Block(
                dim=embed_dim, num_heads=num_heads_in_last_block, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[-1], norm_layer=norm_layer)])
        self.norm = norm_layer(embed_dim, eps=1e-6)
        # self.adapter = ViTAdapter(embed_dim, 1024, 16)
        self.out_type = out_type
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices
        self.out_scales = out_scales

        self.apply(self._init_weights)

        self.frozen_stages = frozen_stages
        if self.frozen_stages > 0:
            self._freeze_stages()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if (not self.with_cls_token):
                # and ckpt_pos_embed_shape[1] == self.pos_embed.shape[1] + 1):
            # Remove cls token from state dict if it's not used.
            state_dict[name] = state_dict[name][:, 1:]
            ckpt_pos_embed_shape = state_dict[name].shape

        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.')

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            ckpt_pos_embed_shape = (16, 8)
            pos_embed_shape = self.patch_embed.init_out_size
            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                'bicubic',
                                                self.num_extra_tokens)

            
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
            param_dict = {k.replace("backbone.", ""): v for k, v in param_dict.items()}
            param_dict = {k.replace("image_encoder.", ""): v for k, v in param_dict.items()}
            param_dict = {k.replace("module.", ""): v for k, v in param_dict.items()}
            param_dict = {k.replace("student.", ""): v for k, v in param_dict.items()}

            self._prepare_pos_embed(param_dict, '')
            msg = self.load_state_dict(param_dict, strict=False)
            print('load vit:', msg)
            # param_dict = {k.replace("projection.", "proj."): v for k, v in param_dict.items()}
        #     count = 0
        #     model_keys = self.state_dict().keys()
        #     for k, v in param_dict.items():
        #         if 'head' in k or 'dist' in k or 'pre_logits' in k or 'decoder' in k or 'inter' in k or 'mask' in k:
        #             continue
        #         if k not in model_keys:
        #             continue
                
        #         if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
        #             # For old models that I trained prior to conv based patchification
        #             O, I, H, W = self.patch_embed.proj.weight.shape
        #             v = v.reshape(O, -1, H, W)
        #         elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
        #             # To resize pos embedding when using model at different size from pretrained weights
        #             dist_shape = (self.pos_embed.shape[1] - self.num_extra_tokens, 1)
        #             src_shape = (v.shape[1] - 1, 1) 
        #             # print('pose embe:',v.shape)
        #             if not self.with_cls_token:
        #                 v = v[:,1:,:]
        #             # print(src_shape, dist_shape)
        #             v = resize_pos_embed(v, src_shape, dist_shape, num_extra_tokens=self.num_extra_tokens)
        #         try:
        #             self.state_dict()[k].copy_(v)
        #             count +=1
        #         except:
        #             print('===========================ERROR=========================')
        #             print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
        #     print('Load %d / %d layers.'%(count,len(self.state_dict().keys())))
            
        #     n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #     print('number of params (M): %.2f' % (n_parameters / 1.e6))
            
        else:
            super(VisionTransformer, self).init_weights()

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze cls_token
        if self.cls_token is not None:
            self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.blocks):
            
            self.norm.eval()
            for param in self.norm.parameters():
                param.requires_grad = False


    def interpolate_pos_encoding_v1(self, x, h, w):
        npatch = x.shape[1] - 1 if self.with_cls_token else x.shape[1]
        N = self.pos_embed.shape[1] - 1 if self.with_cls_token else self.pos_embed.shape[1]
        if npatch == N and self.img_size == (h,w):
            return self.pos_embed
        
        class_pos_embed = self.pos_embed[:, 0] if self.with_cls_token else None
        patch_pos_embed = self.pos_embed[:, 1:] if self.with_cls_token else self.pos_embed
        dim = x.shape[-1]
        OH = self.img_size[0] // self.patch_embed.patch_size
        OW = self.img_size[1] // self.patch_embed.patch_size
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, OH, OW, dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / OH, w0 / OW),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-1] and int(h0) == patch_pos_embed.shape[-2]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1) if self.with_cls_token else patch_pos_embed

    def interpolate_pos_encoding(self, x, pos_embed, npatch, with_cls_token=True):
        # npatch = x.shape[1] - 1 if with_cls_token else x.shape[1]
        N = pos_embed.shape[1] - 1 if with_cls_token else pos_embed.shape[1]
        if npatch == N:
            return pos_embed
        if with_cls_token:
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
        else:
            patch_pos_embed = pos_embed
        dim = x.shape[-1]
        #  h0 = h // self.patch_embed.patch_size
        #  w0 = w // self.patch_embed.patch_size
        #  w0, h0 = w0 + 0.1, h0 + 0.1
        L = npatch
        
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, N, 1, dim).permute(0, 3, 1, 2),
            size=(L, 1),
            # scale_factor=(L / N, 1),
            mode='bicubic',
        )
        assert L == patch_pos_embed.shape[2],"{} vs {}".format(L, patch_pos_embed.shape[2])
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        if with_cls_token:
            patch_pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        return patch_pos_embed
    
    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        # print('x',x.shape)
        x, patch_resolution = self.patch_embed(x)  # patch linear embedding
        npatch = x.shape[1]
        # print('x1',x.shape)

        # add the [CLS] token to the embed patch tokens
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
        # add positional encoding to each token
        # x = x + self.interpolate_pos_encoding(x, w, h)
        # x = x + resize_pos_embed(
        #     self.pos_embed,
        #     self.patch_resolution,
        #     patch_resolution,
        #     mode='bicubic',
        #     num_extra_tokens=self.num_extra_tokens)
        x = x + self.interpolate_pos_encoding(x, self.pos_embed, npatch, self.with_cls_token)
        return self.pos_drop(x), patch_resolution

    def forward(self, x):
        x, patch_resolution = self.prepare_tokens(x)
        outs = []
        
        for i,blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        # x = self.adapter(x)
        x = self._format_output(x, patch_resolution)
        
        return tuple([x])
    
    def _format_output(self, x, hw):
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            return x[:, 0]

        patch_token = x[:, self.num_extra_tokens:]
        if self.out_type == 'featmap':
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2).contiguous()
        if self.out_type == 'avg_featmap':
            return self.ln2(patch_token.mean(dim=1))
        
    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x, _ = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

class AlignDecoder(nn.Module):
    def __init__(self, embed_dim, out_dim=768, nlayers=3, hidden_dim=2048, bottleneck_dim=256, **args):
        super().__init__()
        
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(embed_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(embed_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        self.last_layer = nn.Linear(bottleneck_dim, out_dim)
        self.decoder_norm = nn.LayerNorm(out_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.mlp(x)
        x = self.decoder_norm(self.last_layer(x))
        return x
    
class ViTAdapter(nn.Module):
    def __init__(self, embed_dim, align_dim=768, align_num_heads=12, nlayers=3, hidden_dim=1024, bottleneck_dim=256, **args):
        super().__init__()
        
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.neck = nn.Linear(embed_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(embed_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        layers.append(nn.Linear(bottleneck_dim, align_dim))
        layers.append(nn.LayerNorm(align_dim, eps=1e-6))
        self.neck = nn.Sequential(*layers)
        
        self.align_block = Block(dim=align_dim, num_heads=align_num_heads, mlp_ratio=4, qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.align_norm = nn.LayerNorm(align_dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.neck(x)
        x = self.align_block(x)
        x = self.align_norm(x)
        return x
    
def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

