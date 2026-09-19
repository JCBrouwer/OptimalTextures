"""
Score how well a synthesized texture matches its style: the sliced Wasserstein distance between the VGG features of
both images, per layer. Lower is better. This is the quantity Optimal Textures minimizes, measured on the decoded
image rather than on the features, so it also captures what the decoders lose.

python evaluate.py --style style/graffiti.jpg --size 512 output/*.png
"""

import argparse

import torch

from histmatch import quantiles
from util import load_image, load_styles
from vgg import Encoder


@torch.inference_mode()
def sliced_wasserstein(a: torch.Tensor, b: torch.Tensor, projections: int = 128, seed: int = 0):
    """a, b: [n, c] and [m, c] point clouds -> mean squared 1D Wasserstein-2 distance over random directions"""
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn(a.shape[1], projections, generator=generator)
    directions = (directions / directions.norm(dim=0, keepdim=True)).to(a)
    pa, pb = (a @ directions).T.sort(dim=1).values, (b @ directions).T.sort(dim=1).values
    n = min(pa.shape[1], pb.shape[1], 4096)
    return (quantiles(pa, n) - quantiles(pb, n)).square().mean()


@torch.inference_mode()
def score(image: torch.Tensor, style: torch.Tensor, encoder: Encoder):
    scores = []
    for fs, fi in zip(encoder.pyramid(style), encoder.pyramid(image)):
        fs, fi = fs.flatten(0, 2), fi.flatten(0, 2)
        scores.append((sliced_wasserstein(fi, fs) / fs.var(dim=0).mean()).item())  # relative to the style's spread
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--style", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--style_scale", type=float, default=1.0)
    args = parser.parse_args()

    encoder = Encoder()
    style = load_styles([args.style], args.size, args.style_scale)[0]
    print(f"{'relu1_1':>8}{'relu2_1':>8}{'relu3_1':>8}{'relu4_1':>8}{'relu5_1':>8}{'mean':>8}")
    for path in args.images:
        image = load_image(path, args.size)
        scores = score(image, style, encoder)
        print("".join(f"{s:8.4f}" for s in scores + [sum(scores) / len(scores)]), path)
