import argparse
from functools import partial

import torch
from kornia.color.hls import hls_to_rgb, rgb_to_hls
from torch import Tensor
from torch.nn.functional import interpolate

from histmatch import hist_match
from util import load_styles, maybe_load_content, round32, save_image, to_nchw, to_nhwc, get_size
from vgg import Decoder, Encoder
from typing import List, Optional

downsample = partial(interpolate, mode="nearest")  # for mixing mask
resize = partial(interpolate, mode="bicubic", align_corners=False, antialias=True)


class OptimalTexture(torch.nn.Module):
    def __init__(
        self,
        size: int = 512,
        iters: int = 500,
        passes: int = 5,
        hist_mode: str = "chol",
        color_transfer: Optional[str] = None,
        content_strength: float = 0.1,
        style_scale: float = 1,
        mixing_alpha: float = 0.5,
        no_pca: bool = False,
        no_multires: bool = False,
    ):
        super().__init__()

        self.hist_mode = hist_mode
        self.color_transfer = color_transfer
        self.content_strength = content_strength
        self.style_scale = style_scale
        self.mixing_alpha = mixing_alpha
        self.use_pca = not no_pca
        self.use_multires = not no_multires

        # get number of iterations and sizes for optization
        self.passes = passes
        self.iters_per_pass_and_layer, self.sizes = get_iters_and_sizes(size, iters, passes, self.use_multires)

        self.encoders = torch.nn.ModuleList([Encoder(l) for l in range(1, 6)])
        self.decoders = torch.nn.ModuleList([Decoder(l) for l in range(1, 6)])

    def forward(
        self,
        pastiche: Tensor,
        styles: List[Tensor],
        content: Optional[Tensor] = None,
        verbose: bool = False,
    ):
        for p in range(self.passes):
            if verbose:
                print(f"Pass {p}, size {self.sizes[p]}")

            if self.use_multires:
                style_tens = [
                    resize(s, size=get_size(self.sizes[p], self.style_scale, s.shape[2], s.shape[3])) for s in styles
                ]
                if content is not None:
                    cont_size = get_size(self.sizes[p], 1.0, content.shape[2], content.shape[3], oversize=True)
                    cont_tens = resize(content, size=cont_size)
                else:
                    cont_tens = None
                pastiche = resize(
                    pastiche, size=content.shape[2:] if content is not None else (self.sizes[p], self.sizes[p])
                )

                # get style and content target features
                style_features, style_eigvs, content_features = encode_inputs(
                    self.encoders, style_tens, cont_tens, use_pca=self.use_pca
                )

                if len(styles) > 1:
                    mixing_mask = torch.ceil(
                        torch.rand(style_features[1].shape[1:3], device=pastiche.device) - self.mixing_alpha
                    )[None, None, ...]
                    style_features = mix_style_features(style_features, mixing_mask, self.mixing_alpha, self.hist_mode)

            for l in range(4, -1, -1):
                if verbose:
                    print(f"Layer: relu{l + 1}_1")

                pastiche_feature = self.encoders[l](pastiche)  # encode layer to VGG feature space

                if self.use_pca:
                    pastiche_feature = pastiche_feature @ style_eigvs[l]  # project onto principal components

                for _ in range(self.iters_per_pass_and_layer[p, l - 1]):
                    pastiche_feature = optimal_transport(pastiche_feature, style_features[l], self.hist_mode)

                    if len(content_features) > 0 and l >= 2:  # apply content matching step
                        strength = self.content_strength / 2 ** (4 - l)  # 1, 2, or 4 depending on feature depth
                        pastiche_feature += strength * (content_features[l] - pastiche_feature)

                if self.use_pca:
                    pastiche_feature = pastiche_feature @ style_eigvs[l].T  # reverse principal component projection

                pastiche = self.decoders[l](pastiche_feature)  # decode back to image space

        if self.color_transfer is not None:
            target_hls = rgb_to_hls(content)
            target_hls[:, 1] = rgb_to_hls(pastiche)[:, 1]  # swap lightness channel
            target = hls_to_rgb(target_hls)

            if self.color_transfer == "opt":
                pastiche, target = to_nhwc(pastiche), to_nhwc(target)
                for _ in range(3):
                    pastiche = optimal_transport(pastiche, target, "cdf")
                pastiche = to_nchw(pastiche)

            elif self.color_transfer == "lum":
                pastiche = target  # return pastiche with hue and saturation from content

        return pastiche


def random_rotation(N: int, device: torch.device):
    """
    Draws random N-dimensional rotation matrix (det = 1, inverse = transpose) from the special orthogonal group

    From https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/stats/_multivariate.py#L3309
    """
    H = torch.eye(N, device=device)
    D = torch.empty((N,), device=device)
    for n in range(N - 1):
        x = torch.randn(N - n, device=device)
        norm2 = x @ x
        x0 = x[0].clone()
        D[n] = torch.sign(torch.sign(x[0]) + 0.5)
        x[0] += D[n] * torch.sqrt(norm2)
        x /= torch.sqrt((norm2 - x0**2 + x[0] ** 2) / 2.0)
        H[:, n:] -= torch.outer(H[:, n:] @ x, x)
    D[-1] = (-1) ** (N - 1) * D[:-1].prod()
    H = (D * H.T).T
    return H


def optimal_transport(pastiche_feature: Tensor, style_feature: Tensor, hist_mode: str):
    rotation = random_rotation(pastiche_feature.shape[-1], pastiche_feature.device)

    rotated_pastiche = pastiche_feature @ rotation
    rotated_style = style_feature @ rotation

    matched_pastiche = hist_match(rotated_pastiche, rotated_style, mode=hist_mode)

    pastiche_feature = matched_pastiche @ rotation.T  # rotate back to normal

    return pastiche_feature


def encode_inputs(encoders, styles, content, use_pca):
    style_features, style_eigvs, content_features = [], [], []

    for l in range(5):
        style_features.append(torch.cat([encoders[l](style) for style in styles]))  # encode styles

        if use_pca:
            style_features[l], eigvecs = fit_pca(style_features[l])  # PCA
            style_eigvs.append(eigvecs)

        if content is not None:
            content_feature = encoders[l](content)
            if use_pca:  # project into style PC space
                content_feature = content_feature @ eigvecs
            # center features at mean of style features
            content_feature = content_feature - content_feature.mean() + torch.mean(style_features[l])
            content_features.append(content_feature)

    return style_features, style_eigvs, content_features


def fit_pca(tensor):
    # fit pca
    A = tensor.reshape(-1, tensor.shape[-1]) - tensor.mean()
    _, eigvals, eigvecs = torch.svd(A)
    k = (torch.cumsum(eigvals / torch.sum(eigvals), dim=0) > 0.9).max(0).indices.squeeze()
    eigvecs = eigvecs[:, :k]  # the vectors for 90% of variance will be kept

    # apply to input
    features = tensor @ eigvecs

    return features, eigvecs


def mix_style_features(style_features, mixing_mask, mixing_alpha, hist_mode):
    i = mixing_alpha
    for l, sl in enumerate(style_features):
        mix = to_nhwc(downsample(mixing_mask, size=sl.shape[1:3]))

        A, B = sl[[0]], sl[[1]]
        AtoB = hist_match(A, B, mode=hist_mode)
        BtoA = hist_match(B, A, mode=hist_mode)

        style_target = (A * (1 - i) + AtoB * i) * mix + (BtoA * (1 - i) + B * i) * (1 - mix)

        style_features[l] = style_target
    return style_features


def get_iters_and_sizes(size, iters, passes, use_multires):
    # more iterations for smaller sizes and deeper layers

    if use_multires:
        iters_per_pass = torch.arange(2 * passes, passes, -1)
        iters_per_pass = iters_per_pass / torch.sum(iters_per_pass) * iters

        sizes = torch.linspace(256, size, passes)
        # round to nearest multiple of 32, so that even after 4 max pools the resolution is an even number
        sizes = [round32(size) for size in sizes]
    else:
        iters_per_pass = torch.ones(passes) * int(iters / passes)
        sizes = [size] * passes

    proportion_per_layer = torch.tensor([64, 128, 256, 512, 512]) + 64
    proportion_per_layer = proportion_per_layer / torch.sum(proportion_per_layer)
    iters = (iters_per_pass[:, None] * proportion_per_layer[None, :]).long()

    return iters, sizes


if __name__ == "__main__":

    def required_length(nmin, nmax):
        class RequiredLength(argparse.Action):
            def __call__(self, parser, args, values, option_string=None):
                if not nmin <= len(values) <= nmax:
                    msg = f'argument "{self.dest}" requires between {nmin} and {nmax} arguments'
                    raise argparse.ArgumentTypeError(msg)
                setattr(args, self.dest, values)

        return RequiredLength

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--style", type=str, nargs="+", action=required_length(1, 2), default=["style/graffiti.jpg"], help="Example(s) of the style your texture should take")
    parser.add_argument("-c", "--content", type=str, default=None, help="The structure/shape you want your image to take")
    parser.add_argument("--batch", type=int, default=1, help="Batch size of images to generate")
    parser.add_argument("--size", type=int, default=512, help="The output size of the image (larger output = more memory/time required)")
    parser.add_argument("--passes", type=int, default=5, help="Number of times to loop over each of the 5 layers in VGG-19")
    parser.add_argument("--iters", type=int, default=500, help="Total number of iterations to optimize.")
    parser.add_argument("--hist_mode", type=str, choices=["sym", "pca", "chol", "cdf"], default="chol", help="Histogram matching strategy. CDF is slower than the others, but may use less memory. Each gives slightly different results.")
    parser.add_argument("--color_transfer", type=str, default=None, choices=["lum", "opt"], help="Strategy to employ to keep original color of content image.")
    parser.add_argument("--content_strength", type=float, default=0.01, help="Strength with which to focus on the structure in your content image.")
    parser.add_argument("--style_scale", type=float, default=1.0, help="Scale the style relative to the generated image. Will affect the scale of details generated.")
    parser.add_argument("--mixing_alpha", type=float, default=0.5, help="Value between 0 and 1 for interpolation between 2 textures")
    parser.add_argument("--no_pca", action="store_true", help="Disable PCA of features (slower).")
    parser.add_argument("--no_multires", action="store_true", help="Disable multi-scale rendering (slower, less long-range texture qualities).")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random number generator.")
    parser.add_argument("--no_tf32", action="store_true", help="Disable tf32 format (probably slower).")
    parser.add_argument("--cudnn_benchmark", action="store_true", help="Enable CUDNN benchmarking (probably slower unless doing a lot of iterations).")
    parser.add_argument("--compile", action="store_true", help="Use PyTorch 2.0 compile function to optimize the model.")
    parser.add_argument("--script", action="store_true", help="Use PyTorch JIT script function to optimize the model.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Which device to run on.")
    parser.add_argument("--memory_format", type=str, default="contiguous", choices=["contiguous", "channels_last"], help="Which memory format to use for optimization.")
    parser.add_argument("--output_dir", type=str, default="output/", help="Directory to output results.")
    args = parser.parse_args()
    # fmt: on

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = not args.no_tf32
    torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
    memory_format = torch.contiguous_format if args.memory_format == "contiguous" else torch.channels_last

    if args.seed is not None:
        torch.manual_seed(args.seed)

    with torch.inference_mode():
        styles = load_styles(
            args.style, size=args.size, scale=args.style_scale, device=args.device, memory_format=memory_format
        )
        if len(styles) > 1:
            assert styles[0].shape == styles[1].shape, "Style images must have the same shape"
        content = maybe_load_content(args.content, size=args.size, device=args.device, memory_format=memory_format)
        pastiche = torch.rand(
            content.shape if content is not None else (args.batch, 3, args.size, args.size), device=args.device
        )

        texturizer = OptimalTexture(
            args.size,
            args.iters,
            args.passes,
            args.hist_mode,
            args.color_transfer,
            args.content_strength,
            args.style_scale,
            args.mixing_alpha,
            args.no_pca,
            args.no_multires,
        ).to(pastiche)

        if args.compile:
            texturizer = torch.compile(texturizer)
        if args.script:
            texturizer = torch.jit.script(texturizer)

        from time import time

        t = time()
        pastiche = texturizer.forward(pastiche, styles, content, verbose=True)
        print("Took:", time() - t)

    save_image(pastiche, args)
