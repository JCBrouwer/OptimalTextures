from argparse import Namespace
from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as transforms
from PIL import Image
from torch import Tensor
from torch.nn.functional import interpolate


def load_styles(style_files, size, scale, oversize=False, device="cpu", memory_format=torch.contiguous_format):
    styles = []
    for style_file in style_files:
        styles.append(load_image(style_file, size, scale, not oversize, device=device, memory_format=memory_format))
    return styles


def maybe_load_content(content_file, size, device="cpu", memory_format=torch.contiguous_format):
    content = None
    if content_file is not None:
        content = load_image(content_file, size, oversize=False, device=device, memory_format=memory_format)
    return content


def load_image(path, size, scale=1, oversize=True, device="cpu", memory_format=torch.contiguous_format):
    img = Image.open(path).convert(mode="RGB")
    img = img.resize(get_size(size, scale, img.size[0], img.size[1], oversize), Image.ANTIALIAS)
    return transforms.to_tensor(img).unsqueeze(0).to(device, memory_format=memory_format)


def get_size(size: int, scale: float, h: int, w: int, oversize: bool = False):
    ssize = size * scale
    wpercent = ssize / float(h)
    hsize = int((float(w) * float(wpercent)))

    if oversize:
        size = min(int(ssize), h)
        hsize = min(hsize, w)

    return round32(size), round32(hsize)


def save_image(output: Tensor, args: Namespace):
    outs = [name(style) for style in args.style]
    if len(args.style) > 1:
        outs += ["blend" + str(args.mixing_alpha)]
    if args.content is not None:
        outs += [name(args.content), "strength" + str(args.content_strength)]
    outs += [args.hist_mode + "hist"]
    if args.no_pca:
        outs += ["no_pca"]
    if args.no_multires:
        outs += ["no_multires"]
    if args.style_scale != 1:
        outs += ["scale" + str(args.style_scale)]
    if args.color_transfer is not None:
        outs += [args.color_transfer]
    outs += [str(args.size)]
    outname = "_".join(outs)
    for o, out in enumerate(output):
        torchvision.utils.save_image(
            out, f"{args.output_dir}/{outname}" + (f"_{o + 1}" if len(output) > 1 else "") + ".png"
        )


def get_iters_and_sizes(size: int, iters: int, passes: int, use_multires: bool):
    # more iterations for smaller sizes and deeper layers

    if use_multires:
        iters_per_pass = np.arange(2 * passes, passes, -1)
        iters_per_pass = iters_per_pass / np.sum(iters_per_pass) * iters

        sizes = np.linspace(256, size, passes)
        # round to nearest multiple of 32, so that even after 4 max pools the resolution is an even number
        sizes = (32 * np.round(sizes / 32)).astype(np.int32)
    else:
        iters_per_pass = np.ones(passes) * int(iters / passes)
        sizes = [size] * passes

    proportion_per_layer = np.array([64, 128, 256, 512, 512]) + 64
    proportion_per_layer = proportion_per_layer / np.sum(proportion_per_layer)
    iters = (iters_per_pass[:, None] * proportion_per_layer[None, :]).astype(np.int32)

    return iters.tolist(), sizes.tolist()


def name(filepath: str):
    return filepath.split("/")[-1].split(".")[0]


def round32(integer: int):
    return int(integer + 32 - 1) & -32


def to_nchw(x: Tensor):
    return x.permute(0, 3, 1, 2)


def to_nhwc(x: Tensor):
    return x.permute(0, 2, 3, 1)


def resize(x: Tensor, size: Tuple[int, int]):
    return interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True)
