from __future__ import annotations
from lvsm_pytorch.tensor_typing import Float, Int

from functools import wraps

import torchvision

import torch
from torch import nn, is_tensor
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
    return einx.less('n, b -> b n', seq, lens)

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

def create_and_init_embed(shape):
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

        self.width_embed = create_and_init_embed((max_image_size // patch_size, dim))
        self.height_embed = create_and_init_embed((max_image_size // patch_size, dim))
        self.input_image_embed = create_and_init_embed((max_input_images, dim))

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

        self.mask_ray_embed = create_and_init_embed(dim)
        self.mask_image_embed = create_and_init_embed(dim)

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
        num_images: Int['b'] | None = None
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

# improvised masked autoencoder class

def get_mask_subset_prob(
    mask: Float['b n'],
    prob: float | Float['b'],
    min_mask: int = 0,
    min_keep_mask: int = 0
):
    batch, seq, device = *mask.shape, mask.device

    if is_tensor(prob):
        prob = rearrange(prob, 'b -> b 1')

    total = mask.sum(dim = -1, keepdim = True)

    max_mask = (total - min_keep_mask).clamp(min = 0)

    num_to_mask = (total * prob).long().clamp(min = min_mask)
    num_to_mask = torch.minimum(num_to_mask, max_mask)

    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

class MAE(Module):
    def __init__(
        self,
        lvsm: LVSM,
        frac_masked = 0.5,                  # 1 in 2 image/ray pair to be masked out. minimum set to 1
        frac_images_to_ray_masked = 0.5,    # for a given image/ray pair that is masked, the proportion of images being masked vs rays (1. would be only images masked, 0. would be only rays masked). they cannot be both masked
        image_to_ray_loss_weight = 1.
    ):
        super().__init__()

        self.lvsm = lvsm
        dim = lvsm.dim
        patch_size = lvsm.patch_size

        assert 0. < frac_masked < 1.
        assert 0. < frac_images_to_ray_masked < 1.

        self.frac_masked = frac_masked
        self.frac_images_to_ray_masked = frac_images_to_ray_masked

        self.unpatchify_to_rays = nn.Sequential(
            nn.Linear(dim, 6 * patch_size ** 2),
            Rearrange('... h w (c p1 p2) -> ... c (h p1) (w p2)', p1 = patch_size, p2 = patch_size, c = 6)
        )

        self._c = lvsm._c

        # loss related

        self.image_to_ray_loss_weight = image_to_ray_loss_weight

    def forward(
        self,
        images: Float['b i {self._c} h w'],
        rays: Float['b i 6 h w'],
        num_images: Int['b'] | None = None,
        return_image_and_ray_recon = False,
        return_loss_breakdown = False
    ):
        batch, image_ray_pairs, device = *images.shape[:2], images.device

        # first get the full mask - True means sample exists

        if not exists(num_images):
            mask = torch.ones((batch, image_ray_pairs), device = device, dtype = torch.bool)
        else:
            mask = lens_to_mask(num_images, image_ray_pairs)

        assert (mask.sum(dim = -1) > 1).all(), 'need to have at least 2 image / ray to do self supervised learning'

        # get the images / rays to be masked

        image_ray_mask = get_mask_subset_prob(mask, self.frac_masked, min_mask = 1)

        # then determine the image mask vs the ray mask

        image_mask = get_mask_subset_prob(image_ray_mask, self.frac_images_to_ray_masked)
        ray_mask = image_ray_mask & ~image_mask

        # attention is all you need for 3d understanding w/ overparameterized plucker ray rep

        tokens = self.lvsm.image_and_ray_encoder(
            images = images,
            rays = rays,
            image_mask = image_mask,
            ray_mask = ray_mask,
            num_images = num_images
        )

        # determine loss

        pred_rays = self.unpatchify_to_rays(tokens[ray_mask])
        pred_images = self.lvsm.unpatchify_to_image(tokens[image_mask])

        image_recon_loss = F.mse_loss(
            images[image_mask],
            pred_images,
            reduction = 'none'
        )

        ray_recon_loss = F.mse_loss(
            rays[ray_mask],
            pred_rays,
            reduction = 'none'
        )

        total_loss = torch.cat((image_recon_loss.flatten() * self.image_to_ray_loss_weight, ray_recon_loss.flatten())).mean()

        loss_breakdown = (image_recon_loss, ray_recon_loss)

        recons = ((image_mask, pred_images), (ray_mask, pred_rays))

        if not return_loss_breakdown and not return_image_and_ray_recon:
            return total_loss

        if return_loss_breakdown and not return_image_and_ray_recon:
            return total_loss, loss_breakdown

        if not return_loss_breakdown and return_image_and_ray_recon:
            return total_loss, recons

        return total_loss, (loss_breakdown, recons)

# main class

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

        self.dim = dim
        self.patch_size = patch_size
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

        self.unpatchify_to_image = nn.Sequential(
            nn.Linear(dim, channels * patch_size_sq),
            nn.Sigmoid(),
            Rearrange('... h w (c p1 p2) -> ... c (h p1) (w p2)', p1 = patch_size, p2 = patch_size, c = channels)
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
        return_loss_breakdown = False,
        return_embed = False
    ):
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

        target_tokens, input_tokens = unpack_input_target(tokens)

        # allow for returning embeddings (which should contain rich geometric information)

        if return_embed:
            return target_tokens, input_tokens

        # project back to image

        pred_target_images = self.unpatchify_to_image(target_tokens)

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
    intrinsic_rotation: Float['... 3 3'],
    extrinsic_rotation: Float['... 3 3'],
    translation: Float['... 3'],
    uniform_points: Float['... 3 h w'],
) -> Float['... 6 h w']:

    K_inv = torch.linalg.inv(intrinsic_rotation)

    direction = einsum(extrinsic_rotation, K_inv, uniform_points, '... c1 c2, ... c1 c0, ... c0 h w -> ... c2 h w')
    points = einsum(-extrinsic_rotation, translation, '... c1 c2, ... c1 -> ... c2')

    moments = torch.cross(
        rearrange(points, '... c -> ... c 1 1'),
        direction,
        dim = -3
    )

    return torch.cat((direction, moments), dim = -3)

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

class MAECameraWrapper(Module):
    def __init__(
        self,
        mae: MAE
    ):
        super().__init__()
        self.mae = mae

        # tensor typing

        self._c = mae._c

    def forward(
        self,
        intrinsic_rotation: Float['b i 3 3'],
        extrinsic_rotation: Float['b i 3 3'],
        translation: Float['b i 3'],
        uniform_points: Float['b i 3 h w'],
        images: Float['b i {self._c} h w'],
        num_images: Int['b'] | None = None,
        **kwargs
    ):

        rays = to_plucker_rays(
            intrinsic_rotation,
            extrinsic_rotation,
            translation,
            uniform_points
        )

        out = self.mae(
            images = images,
            rays = rays,
            num_images = num_images,
            **kwargs
        )

        return out
