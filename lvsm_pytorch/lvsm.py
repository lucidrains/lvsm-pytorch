from __future__ import annotations
from lvsm_pytorch.tensor_typing import Float, Int

from functools import wraps

import torchvision

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from x_transformers import Encoder

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, pack, unpack

"""
ein notation:
b - batch
n - sequence
h - height
w - width
c - channels (either 6 for plucker rays or 3 for rgb)
i - input images
"""

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def lens_to_mask(lens: Int['b'], max_length: int):
    seq = torch.arange(max_length, device = lens.device)
    return einx.less('b, n -> b n', lens, seq)

def divisible_by(num, den):
    return (num % den) == 0

def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, '_vgg')
        if has_vgg:
            vgg = self._vgg
            delattr(self, '_vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self._vgg = vgg

        return out
    return inner

# class

class LVSM(Module):
    def __init__(
        self,
        dim,
        *,
        max_image_size,
        patch_size,
        depth = 12,
        heads = 8,
        max_input_images = 32,
        dim_head = 64,
        decoder_kwargs: dict = dict(
            use_rmsnorm = True,
            add_value_residual = True,
            ff_glu = True,
        ),
        perceptual_loss_weight = 0.5    # they use 0.5 for scene-level, 1.0 for object-level
    ):
        super().__init__()
        assert divisible_by(max_image_size, patch_size)

        self.width_embed = nn.Parameter(torch.zeros(max_image_size // patch_size, dim))
        self.height_embed = nn.Parameter(torch.zeros(max_image_size // patch_size, dim))
        self.input_image_embed = nn.Parameter(torch.zeros(max_input_images, dim))

        nn.init.normal_(self.width_embed, std = 0.02)
        nn.init.normal_(self.height_embed, std = 0.02)
        nn.init.normal_(self.input_image_embed, std = 0.02)

        patch_size_sq = patch_size ** 2

        self.input_to_patch_tokens = nn.Sequential(
            Rearrange('b i c (h p1) (w p2) -> b i h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear((6 + 3) * patch_size_sq, dim)
        )

        self.target_rays_to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(6 * patch_size_sq, dim)
        )

        self.decoder = Encoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            **decoder_kwargs
        )

        self.target_unpatchify_to_image = nn.Sequential(
            nn.Linear(dim, 3 * patch_size_sq),
            nn.Sigmoid(),
            Rearrange('b h w (c p1 p2) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size, c = 3)
        )

        self.has_perceptual_loss = perceptual_loss_weight > 0.
        self.perceptual_loss_weight = perceptual_loss_weight

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    @property
    def vgg(self):
        if hasattr(self, '_vgg'):
            return self._vgg

        vgg = torchvision.models.vgg16(pretrained = True)
        vgg.classifier = nn.Sequential(*vgg.classifier[:-2])
        self._vgg = vgg.to(self.device)
        return self._vgg

    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def forward(
        self,
        input_images: Float['b i 3 h w'],
        input_rays: Float['b i 6 h w'],
        target_rays: Float['b 6 h w'],
        target_images: Float['b 3 h w'] | None = None,
        num_input_images: Int['b'] | None = None,
        return_loss_breakdown = False
    ):

        input_tokens = self.input_to_patch_tokens(torch.cat((input_images, input_rays), dim = -3))

        target_tokens = self.target_rays_to_patch_tokens(target_rays)

        # add positional embeddings

        _, num_images, height, width, _ = input_tokens.shape

        input_image_embed = self.input_image_embed[:num_images]
        height_embed = self.height_embed[:height]
        width_embed = self.width_embed[:width]

        input_tokens = einx.add('b i h w d, i d, h d, w d -> b i h w d', input_tokens, input_image_embed, height_embed, width_embed)

        target_tokens = einx.add('b h w d, h d, w d -> b h w d', target_tokens, height_embed, width_embed)

        # pack dimensions to ready for attending

        input_tokens, _ = pack([input_tokens], 'b * d')
        target_tokens, packed_height_width = pack([target_tokens], 'b * d')

        tokens, packed_shape = pack([target_tokens, input_tokens], 'b * d')

        # take care of variable number of input images

        mask = None

        if exists(num_input_images):
            mask = lens_to_mask(num_input_images, num_images + 1) # plus one for target patched rays
            mask = repeat(mask, 'b i -> b (i hw)', hw = height * width)

        # attention

        tokens = self.decoder(tokens, mask = mask)

        # unpack

        target_tokens, input_tokens = unpack(tokens, packed_shape, 'b * d')

        # project target tokens out

        target_tokens, = unpack(target_tokens, packed_height_width, 'b * d')

        # project back to image

        pred_target_images = self.target_unpatchify_to_image(target_tokens)

        if not exists(target_images):
            return pred_target_images

        loss =  F.mse_loss(pred_target_images, target_images)

        perceptual_loss = self.zero

        if self.has_perceptual_loss:
            target_image_vgg_feats = self.vgg(target_images)
            pred_target_image_vgg_feats = self.vgg(pred_target_images)

            perceptual_loss = F.mse_loss(target_image_vgg_feats, pred_target_image_vgg_feats)

        total_loss = (
            loss +
            perceptual_loss * self.perceptual_loss_weight
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, perceptual_loss)
