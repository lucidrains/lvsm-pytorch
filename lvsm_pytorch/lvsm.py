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
from einops import einsum, rearrange, repeat, pack, unpack

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

def pack_with_inverse(t, pattern):
    packed, ps = pack(t, pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpack_pattern = default(unpack_pattern, pattern)
        unpacked = unpack(to_unpack, ps, unpack_pattern)
        return unpacked

    return packed, unpack_one

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
        channels = 3,
        rand_input_image_embed = True,
        dropout_input_ray_prob = 0.,
        decoder_kwargs: dict = dict(
            use_rmsnorm = True,
            add_value_residual = True,
            ff_glu = True,
        ),
        perceptual_loss_weight = 0.5    # they use 0.5 for scene-level, 1.0 for object-level
    ):
        super().__init__()
        assert divisible_by(max_image_size, patch_size)

        # positional embeddings

        self.width_embed = nn.Parameter(torch.zeros(max_image_size // patch_size, dim))
        self.height_embed = nn.Parameter(torch.zeros(max_image_size // patch_size, dim))
        self.input_image_embed = nn.Parameter(torch.zeros(max_input_images, dim))

        nn.init.normal_(self.width_embed, std = 0.02)
        nn.init.normal_(self.height_embed, std = 0.02)
        nn.init.normal_(self.input_image_embed, std = 0.02)

        self.rand_input_image_embed = rand_input_image_embed

        # raw data to patch tokens for attention

        patch_size_sq = patch_size ** 2

        self.images_to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(channels * patch_size_sq, dim)
        )

        self.plucker_rays_to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(6 * patch_size_sq, dim)
        )

        # allow for dropping out input rays
        # to maybe improve transformer 3d understanding from only images

        self.has_dropout_input_ray = dropout_input_ray_prob > 0.
        self.input_ray_dropout = nn.Dropout(dropout_input_ray_prob) # allow for dropping out the input rays

        self.null_ray_embed = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.null_ray_embed, std = 0.02)

        self.null_image_embed = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.null_image_embed, std = 0.02)

        self.decoder = Encoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            **decoder_kwargs
        )

        self.target_unpatchify_to_image = nn.Sequential(
            nn.Linear(dim, channels * patch_size_sq),
            nn.Sigmoid(),
            Rearrange('b h w (c p1 p2) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size, c = channels)
        )

        self.has_perceptual_loss = perceptual_loss_weight > 0. and channels == 3
        self.perceptual_loss_weight = perceptual_loss_weight

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # for tensor typing

        self._c = channels

    @property
    def device(self):
        return self.zero.device

    @property
    def vgg(self):

        if not self.has_perceptual_loss:
            return None

        if hasattr(self, '_vgg'):
            return self._vgg[0]

        vgg = torchvision.models.vgg16(pretrained = True)
        vgg.classifier = nn.Sequential(*vgg.classifier[:-2])
        vgg.requires_grad_(False)

        self._vgg = [vgg]
        return vgg.to(self.device)

    def forward(
        self,
        input_images: Float['b i {self._c} h w'],
        input_rays: Float['b i 6 h w'],
        target_rays: Float['b 6 h w'],
        target_images: Float['b {self._c} h w'] | None = None,
        num_input_images: Int['b'] | None = None,
        return_loss_breakdown = False
    ):

        # get input image tokens

        input_images, unpack_images_batch = pack_with_inverse([input_images], '* c h w')

        input_image_tokens = self.images_to_patch_tokens(input_images)

        input_image_tokens, = unpack_images_batch(input_image_tokens)

        # get both input and target plucker ray based tokens

        rays, unpack_input_target = pack_with_inverse([input_rays, target_rays], '* c h w')

        ray_tokens = self.plucker_rays_to_patch_tokens(rays)

        input_ray_tokens, target_ray_tokens = unpack_input_target(ray_tokens)

        # maybe dropout input rays

        if self.training and self.has_dropout_input_ray:
            ones = input_ray_tokens.new_ones(input_rays.shape[:2])
            dropout_mask = self.input_ray_dropout(ones)

            input_ray_tokens = einx.where(
                'b i, b i h w d, d -> b i h w d',
                dropout_mask > 0., input_ray_tokens, self.null_ray_embed
            )

        # input tokens have summed contribution from image + rays

        input_tokens = input_image_tokens + input_ray_tokens

        target_tokens = target_ray_tokens + self.null_image_embed

        # add positional embeddings

        _, num_images, height, width, _ = input_tokens.shape

        height_embed = self.height_embed[:height]
        width_embed = self.width_embed[:width]

        input_tokens = einx.add('b i h w d, h d, w d -> b i h w d', input_tokens, height_embed, width_embed)

        target_tokens = einx.add('b h w d, h d, w d -> b h w d', target_tokens, height_embed, width_embed)

        # add input image embeddings, make it random to prevent overfitting

        if self.rand_input_image_embed:
            batch, max_num_input_images = input_tokens.shape[0], self.input_image_embed.shape[0]

            randperm = torch.randn((batch, max_num_input_images), device = self.device).argsort(dim = -1)
            randperm = randperm[:, :num_images]

            rand_input_image_embed = self.input_image_embed[randperm]

            input_tokens = einx.add('b i h w d, b i d -> b i h w d', input_tokens, rand_input_image_embed)
        else:
            input_image_embed = self.input_image_embed[:num_images]
            input_tokens = einx.add('b i h w d, i d -> b i h w d', input_tokens, input_image_embed)

        # pack dimensions to ready for attending

        input_tokens, _ = pack([input_tokens], 'b * d')
        target_tokens, unpack_height_width = pack_with_inverse([target_tokens], 'b * d')

        tokens, unpack_target_input_tokens = pack_with_inverse([target_tokens, input_tokens], 'b * d')

        # take care of variable number of input images

        mask = None

        if exists(num_input_images):
            mask = lens_to_mask(num_input_images, num_images + 1) # plus one for target patched rays
            mask = repeat(mask, 'b i -> b (i hw)', hw = height * width)

        # attention

        tokens = self.decoder(tokens, mask = mask)

        # unpack

        target_tokens, input_tokens = unpack_target_input_tokens(tokens)

        # project target tokens out

        target_tokens, = unpack_height_width(target_tokens)

        # project back to image

        pred_target_images = self.target_unpatchify_to_image(target_tokens)

        if not exists(target_images):
            return pred_target_images

        loss =  F.mse_loss(pred_target_images, target_images)

        perceptual_loss = self.zero

        if self.has_perceptual_loss:
            self.vgg.eval()

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

# a wrapper for converting camera in/ex - trinsics into the Plucker 6D representation
# complete noob in this area, but following figure 2. in https://arxiv.org/html/2402.14817v1
# feel free to open an issue if you see some obvious error

def to_plucker_rays(
    intrinsic_rotation: Float['b 3 3'],
    extrinsic_rotation: Float['b 3 3'],
    translation: Float['b 3'],
    uniform_points: Float['b 3 h w'],
) -> Float['b 6 h w']:

    K_inv = torch.linalg.inv(intrinsic_rotation)

    direction = einsum(extrinsic_rotation, K_inv, uniform_points, 'b c1 c2, b c1 c0, b c0 h w -> b c2 h w')
    points = einsum(-extrinsic_rotation, translation, 'b c1 c2, b c1 -> b c2')

    moments = torch.cross(
        rearrange(points, 'b c -> b c 1 1'),
        direction,
        dim = 1
    )

    return torch.cat((direction, moments), dim = 1)

class CameraWrapper(Module):
    def __init__(
        self,
        lvsm: LVSM
    ):
        super().__init__()
        self.lvsm = lvsm

        # tensor typing

        self._c = lvsm._c

    def forward(
        self,
        input_intrinsic_rotation: Float['b i 3 3'],
        input_extrinsic_rotation: Float['b i 3 3'],
        input_translation: Float['b i 3'],
        input_uniform_points: Float['b i 3 h w'],
        target_intrinsic_rotation: Float['b 3 3'],
        target_extrinsic_rotation: Float['b 3 3'],
        target_translation: Float['b 3'],
        target_uniform_points: Float['b 3 h w'],
        input_images: Float['b i {self._c} h w'],
        target_images: Float['b {self._c} h w'] | None = None,
        num_input_images: Int['b'] | None = None,
        return_loss_breakdown = False
    ):

        intrinsic_rotation, packed_shape = pack([input_intrinsic_rotation, target_intrinsic_rotation], '* i j')
        extrinsic_rotation, _ = pack([input_extrinsic_rotation, target_extrinsic_rotation], '* i j')
        translation, _ = pack([input_translation, target_translation], '* j')
        uniform_points, _ = pack([input_uniform_points, target_uniform_points], '* c h w')

        plucker_rays = to_plucker_rays(
            intrinsic_rotation,
            extrinsic_rotation,
            translation,
            uniform_points
        )

        input_rays, target_rays = unpack(plucker_rays, packed_shape, '* c h w')

        out = self.lvsm(
            input_images = input_images,
            input_rays = input_rays,
            target_rays = target_rays,
            target_images = target_images,
            num_input_images = num_input_images,
            return_loss_breakdown = return_loss_breakdown
        )

        return out
