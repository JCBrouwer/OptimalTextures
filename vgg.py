"""
Adapted from https://github.com/pietrocarbo/deep-transfer
"""

import os
from itertools import chain

import torch
import torch.nn as nn

from util import to_nchw, to_nhwc

# lambdas to delay creation of modules until actually needed
vgg_normalized = lambda _: [
    [
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),
    ],
    # ^ conv1_1
    [
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),
    ],
    # ^ conv2_1
    [
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),
    ],
    # ^ conv3_1
    [
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),
    ],
    # ^ conv4_1
    [
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
    ],
    # ^ conv5_1
]


feature_invertor = lambda _: [
    [
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
    ],
    # ^ conv5_1
    [
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
    ],
    # ^ conv4_1
    [
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
    ],
    # ^ conv3_1
    [
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
    ],
    # ^ conv2_1
    [
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    ],
    # ^ conv1_1
]


class Encoder(nn.Module):
    """
    Normalized VGG-19 up to relu5_1, split into the five stages that end at relu1_1 ... relu5_1. The shallower
    encoders are prefixes of the deepest one, so a single set of weights serves every depth.
    """

    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList(nn.Sequential(*stage) for stage in vgg_normalized(None))
        state_dict = torch.load(f"{os.path.dirname(__file__)}/models/vgg_normalised_conv5_1.pth")
        flat = nn.Sequential(*chain.from_iterable(self.stages))  # the checkpoint indexes layers without stages
        flat.load_state_dict(state_dict)

    def forward(self, x, depth: int = 5):
        """Features at relu{depth}_1 -> NHWC so that matmuls with PCA and rotations are easier"""
        for stage in self.stages[:depth]:
            x = stage(x)
        return to_nhwc(x)

    def pyramid(self, x):
        """Features at relu1_1 ... relu5_1 from a single pass"""
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(to_nhwc(x))
        return features


class Decoder(nn.Module):
    def __init__(self, depth):
        super().__init__()
        assert isinstance(depth, int) and 1 <= depth <= 5
        self.depth = depth
        self.model = nn.Sequential(*chain.from_iterable(feature_invertor(None)[-depth:]))
        self.model.load_state_dict(torch.load(f"{os.path.dirname(__file__)}/models/feature_invertor_conv{depth}_1.pth"))

    def forward(self, x):
        return self.model(to_nchw(x))
