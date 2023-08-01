import torch
import torch.nn as nn

import os
import sys

from ml.extra_libraries.pytorch_ssim import SSIM3D
from ml.extra_libraries.cldice import soft_cldice, soft_dice_cldice

def minimax_norm(tensor):
    normed = (tensor - tensor.min())/(tensor.max() - tensor.min())
    return normed


class CycleLoss:
    def __init__(self, type='mse'):
        self.loss_fn = nn.MSELoss()

    def __call__(self, real, cycled):
        return self.loss_fn(real, cycled) 


class ReconstructionLoss:
    def __init__(self):
        self.loss_fn = SSIM3D()

    def __call__(self, real, cycled):
        return self.loss_fn(real, cycled)


class SegmentationLoss:
    def __init__(self):
        self.loss_fn = soft_cldice()

    def __call__(self, real, cycled):
        return self.loss_fn(real, cycled)


class DiscriminatorLoss:
    def __init__(self, type=None):
        self.loss_fn = nn.MSELoss()

    def __call__(self, real, fake):
        return 0.5 * (self.loss_fn(torch.ones_like(real), real) +
                      self.loss_fn(torch.zeros_like(fake), fake))

class GeneratorLoss:
    def __init__(self, type=None):
        self.loss_fn = nn.MSELoss()

    def __call__(self, dicriminator_response_to_fake):
        return (self.loss_fn(torch.ones_like(dicriminator_response_to_fake), dicriminator_response_to_fake))