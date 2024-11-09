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

def pad_at_dim(t, pad: tuple[int, int], *, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pack_with_inverse(t, pattern):
    packed, ps = pack(t, pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpack_pattern = default(unpack_pattern, pattern)
        unpacked = unpack(to_unpack, ps, unpack_pattern)
        return unpacked

    return packed, unpack_one

def init_embed(shape):
    params = nn.Parameter(torch.zeros(shape))
    nn.init.normal_(params, std = 0.02)
    return params

# plucker ray transformer encoder
# it can accept a mask for either dropping out images or rays for a given sample in a batch
# this is needed to generalize for both supervised and self-supervised learning (MAE from Kaiming He)

class ImageAndPluckerRayEncoder(Module):
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
    ):
        super().__init__()
        assert divisible_by(max_image_size, patch_size)

        # positional embeddings

        self.width_embed = init_embed((max_image_size // patch_size, dim))
        self.height_embed = init_embed((max_image_size // patch_size, dim))
        self.input_image_embed = init_embed((max_input_images, dim))

        self.rand_input_image_embed = rand_input_image_embed

        # raw data to patch tokens for attention

        patch_size_sq = patch_size ** 2

        self.images_to_patch_tokens = nn.Sequential(
            Rearrange('b i c (h p1) (w p2) -> b i h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(channels * patch_size_sq, dim)
        )

        self.plucker_rays_to_patch_tokens = nn.Sequential(
            Rearrange('b i c (h p1) (w p2) -> b i h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(6 * patch_size_sq, dim)
        )

        self.mask_ray_embed = init_embed(dim)
        self.mask_image_embed = init_embed(dim)

        self.decoder = Encoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            **decoder_kwargs
        )

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # for tensor typing

        self._c = channels

    @property
    def device(self):
        return self.zero.device

    def forward(
        self,
        images: Float['b i {self._c} h w'],
        rays: Float['b i 6 h w'],
        image_mask: Bool['b i'] | None = None,
        ray_mask: Bool['b i'] | None = None,
        num_images: Int['b'] | None = None,
        return_loss_breakdown = False
    ):
        # get image tokens

        image_tokens = self.images_to_patch_tokens(images)

        # get ray tokens

        ray_tokens = self.plucker_rays_to_patch_tokens(rays)

        # take care of masking either image or ray tokens

        if exists(image_mask):
            image_tokens = einx.where('b i, d, b i h w d -> b i h w d', image_mask, self.mask_image_embed, image_tokens)

        if exists(ray_mask):
            ray_tokens = einx.where('b i, d, b i h w d -> b i h w d', ray_mask, self.mask_ray_embed, ray_tokens)

        # input tokens have summed contribution from image + rays

        tokens = image_tokens + ray_tokens

        # add positional embeddings

        _, image_ray_pairs, height, width, _ = tokens.shape

        height_embed = self.height_embed[:height]
        width_embed = self.width_embed[:width]

        tokens = einx.add('b i h w d, h d, w d -> b i h w d', tokens, height_embed, width_embed)

        # add input image embeddings, make it random to prevent overfitting

        if self.rand_input_image_embed:
            batch, max_num_images = tokens.shape[0], self.input_image_embed.shape[0]

            randperm = torch.randn((batch, max_num_images), device = self.device).argsort(dim = -1)
            randperm = randperm[:, :image_ray_pairs]

            rand_input_image_embed = self.input_image_embed[randperm]

            tokens = einx.add('b i h w d, b i d -> b i h w d', tokens, rand_input_image_embed)
        else:
            input_image_embed = self.input_image_embed[:image_ray_pairs]
            tokens = einx.add('b i h w d, i d -> b i h w d', tokens, input_image_embed)

        # take care of variable number of input images

        mask = None

        if exists(num_images):
            mask = lens_to_mask(num_images, image_ray_pairs) # plus one for target patched rays
            mask = repeat(mask, 'b i -> b (i hw)', hw = height * width)

        # attention

        tokens, inverse_pack = pack_with_inverse([tokens], 'b * d')

        embed = self.decoder(tokens, mask = mask)

        embed, = inverse_pack(embed)

        return embed

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

        patch_size_sq = patch_size ** 2

        self.input_ray_dropout = nn.Dropout(dropout_input_ray_prob)

        self.image_and_ray_encoder = ImageAndPluckerRayEncoder(
            dim = dim,
            max_image_size = max_image_size,
            patch_size = patch_size,
            depth = depth,
            heads = heads,
            max_input_images = max_input_images,
            dim_head = dim_head,
            channels = channels,
            rand_input_image_embed = rand_input_image_embed,
            decoder_kwargs = decoder_kwargs
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

        # if target images not given, assume inferencing

        is_training = exists(target_images)
        is_inferencing = not is_training

        # ray mask, by default attend using all rays, but this may not be true for MAE

        batch_num_images_shape = input_images.shape[:2]

        ray_mask = torch.zeros(batch_num_images_shape, device = self.device, dtype = torch.bool)
        image_mask = torch.zeros(batch_num_images_shape, device = self.device, dtype = torch.bool)

        # maybe dropout input rays

        dropout_mask = self.input_ray_dropout((~ray_mask).float())
        ray_mask = dropout_mask == 0.

        # target ray will never be masked out

        ray_mask = F.pad(ray_mask, (1, 0), value = False)

        # place the target image and ray at the very left-hand side

        # add a dummy image for the target image being predicted
        # target mask will be set to True

        images = pad_at_dim(input_images, (1, 0), dim = 1)
        image_mask = F.pad(image_mask, (1, 0), value = True)

        # get both input and target plucker ray based tokens

        rays, unpack_input_target = pack_with_inverse([target_rays, input_rays], 'b * c h w')

        # add 1 to num_input_images for target

        if exists(num_input_images):
            num_input_images = num_input_images + 1

        # image and plucker ray encoder

        tokens = self.image_and_ray_encoder(
            images = images,
            rays = rays,
            ray_mask = ray_mask,
            image_mask = image_mask,
            num_images = num_input_images
        )

        # extract target tokens

        target_tokens = tokens[:, 0]

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
