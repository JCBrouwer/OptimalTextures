"""
Adapted from https://github.com/pietrocarbo/deep-transfer
"""

import gc
import os
from itertools import chain

import torch
import torch.nn as nn

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
    def __init__(self, depth):
        super(Encoder, self).__init__()
        assert isinstance(depth, int) and 1 <= depth <= 5
        self.depth = depth
        self.model = nn.Sequential(*chain.from_iterable(vgg_normalized(None)[:depth]))
        self.model.load_state_dict(torch.load(f"{os.path.dirname(__file__)}/models/vgg_normalised_conv{depth}_1.pth"))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        del self
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, x):
        return self.model(x).permute(0, 2, 3, 1)  # -> [b, h, w, c] so that matmuls with PCA and rotations are easier


class Decoder(nn.Module):
    def __init__(self, depth):
        super(Decoder, self).__init__()
        assert isinstance(depth, int) and 1 <= depth <= 5
        self.depth = depth
        self.model = nn.Sequential(*chain.from_iterable(feature_invertor(None)[-depth:]))
        self.model.load_state_dict(torch.load(f"{os.path.dirname(__file__)}/models/feature_invertor_conv{depth}_1.pth"))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        del self
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, x):
        return self.model(x.permute(0, 3, 1, 2))  # -> [b, h, w, c] so that matmuls with PCA and rotations are easier
