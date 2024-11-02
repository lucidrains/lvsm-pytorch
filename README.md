<img src="./lvsm.png" width="500px"></img>

<img src="./lvsm-finding.png" width="400px"></img>

## LVSM - Pytorch (wip)

Implementation of [LVSM](https://haian-jin.github.io/projects/LVSM/), SOTA Large View Synthesis with Minimal 3d Inductive Bias, from Adobe Research

We will focus only on the Decoder-only architecture in this repository.

This paper lines up with <a href="https://openreview.net/forum?id=A8Vuf2e8y6">another</a> from ICLR 2025

## Install

```bash
$ pip install lvsm-pytorch
```

## Usage

```python
import torch
from lvsm_pytorch import LVSM

rays = torch.randn(2, 6, 256, 256)
images = torch.randn(2, 3, 256, 256)

target_rays = torch.randn(2, 6, 256, 256)
target_images = torch.randn(2, 3, 256, 256)

model = LVSM(
    dim = 512,
    patch_size = 32,
    depth = 2,
)

loss = model(
    input_images = images,
    input_rays = rays,
    target_rays = target_rays,
    target_images = target_images
)

loss.backward()

# after much training

pred_images = model(
    input_images = images,
    input_rays = rays,
    target_rays = target_rays,
) # (2, 3, 256, 256)

assert pred_images.shape == target_images.shape
```

## Citations

```bibtex
@inproceedings{Jin2024LVSMAL,
    title   = {LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias},
    author  = {Haian Jin and Hanwen Jiang and Hao Tan and Kai Zhang and Sai Bi and Tianyuan Zhang and Fujun Luan and Noah Snavely and Zexiang Xu},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273507016}
}
```
