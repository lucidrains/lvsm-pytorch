import torch
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from x_transformers import Encoder

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# class

class LVSM(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
