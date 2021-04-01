import argparse
from functools import partial

import matplotlib.pyplot as plt
import torch
from torch.nn.functional import interpolate
from tqdm import tqdm

import util
from histmatch import *
from vgg import Decoder, Encoder

downsample = partial(interpolate, mode="nearest")  # for mixing mask
upsample = partial(interpolate, mode="bicubic", align_corners=False)  # for output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def optimal_texture(
    style,
    content=None,
    size=512,
    content_strength=0.01,
    mixing_alpha=0.5,
    hist_mode="chol",
    no_pca=False,
    no_multires=False,
    passes=5,
    iters=500,
):
    # readability booleans
    use_pca = not no_pca
    use_multires = not no_multires
    texture_mixing = len(style) > 1

    # get number of iterations and sizes for optization
    iters_per_pass_and_layer, sizes = get_iters_and_sizes(size, iters, passes, use_multires)

    # load inputs and initialize output image
    styles = util.load_styles(style, size=sizes[0])
    content = util.maybe_load_content(content, size=sizes[0])
    output = torch.rand((1, 3, sizes[0], sizes[0]), device=device)

    # transform style and content to VGG feature space
    style_layers, style_eigvs, content_layers = encode_inputs(styles, content, use_pca=use_pca)

    if texture_mixing:
        mixing_mask = torch.ceil(torch.rand(style_layers[1].shape[1:3], device=device) - mixing_alpha)[None, None, ...]
        style_layers = mix_style_layers(style_layers, mixing_mask, mixing_alpha, hist_mode)

    pbar = tqdm(total=iters, smoothing=1)
    for p in range(passes):

        if use_multires and p != 0:
            # upsample to next size
            output = upsample(output, size=(sizes[p], sizes[p]))

            # reload style and content at the new size
            styles = util.load_styles(args.style, size=sizes[p])
            content = util.maybe_load_content(args.content, size=sizes[p])
            style_layers, style_eigvs, content_layers = encode_inputs(styles, content, use_pca=use_pca)
            if texture_mixing:
                style_layers = mix_style_layers(style_layers, mixing_mask, mixing_alpha, hist_mode)

        for l in range(5, 0, -1):
            pbar.set_description(f"Current resolution: {sizes[p if use_multires else 0]}, current layer relu{l}_1")

            # encode layer to VGG feature space
            with Encoder(l).to(device) as encoder:
                output_layer = encoder(output)
            if use_pca:
                output_layer = output_layer @ style_eigvs[l]  # project onto principle components

            # iteratively apply optimal transport (rotate randomly and match histograms)
            for _ in range(iters_per_pass_and_layer[p, l - 1]):
                rotation = random_rotation(output_layer.shape[-1])

                rotated_output = output_layer @ rotation
                rotated_style = style_layers[l] @ rotation

                matched_output = hist_match(rotated_output, rotated_style, mode=hist_mode)

                output_layer = matched_output @ rotation.T  # rotate back to normal

                # apply content matching step
                if content is not None and l >= 3:
                    strength = content_strength
                    strength /= 2 ** (5 - l)  # 1, 2, or 4 depending on layer depth
                    output_layer += strength * (content_layers[l] - output_layer)

                pbar.update(1)

            if use_pca:
                output_layer = output_layer @ style_eigvs[l].T  # reverse principle component projection
            with Decoder(l).to(device) as decoder:
                output = decoder(output_layer)  # decode back to image space

    return output


def encode_inputs(styles, content, use_pca):
    style_layers, style_eigvs, content_layers = [None], [None], [None]

    for l in range(1, 6):
        with Encoder(l).to(device) as encoder:
            style_layers.append(torch.cat([encoder(style) for style in styles]))  # encode styles

            if use_pca:
                style_layers[l], eigvecs = fit_pca(style_layers[l])  # PCA
                style_eigvs.append(eigvecs)

            if content is not None:
                content_layer = encoder(content)
                if use_pca:  # project into style PC space
                    content_layer = content_layer @ eigvecs
                # center features at mean of style features
                content_layer = content_layer - content_layer.mean() + torch.mean(style_layers[l])
                content_layers.append(content_layer)

    return style_layers, style_eigvs, content_layers


def fit_pca(tensor):
    # fit pca
    A = tensor.reshape(-1, tensor.shape[-1]) - tensor.mean()
    _, eigvals, eigvecs = torch.svd(A)
    total_variance = torch.sum(eigvals)
    k = np.argmax(np.cumsum([i / total_variance for i in eigvals]) > 0.9)
    eigvecs = eigvecs[:, :k]  # the vectors for 90% of variance will be kept

    # apply to input
    features = tensor @ eigvecs

    return features, eigvecs


def mix_style_layers(style_layers, mixing_mask, mixing_alpha, hist_mode):
    i = mixing_alpha
    for l, sl in enumerate(style_layers[1:]):
        mix = downsample(mixing_mask, size=sl.shape[1:3]).permute(0, 2, 3, 1)

        A, B = sl[[0]], sl[[1]]
        AtoB = hist_match(A, B, mode=hist_mode)
        BtoA = hist_match(B, A, mode=hist_mode)

        style_target = (A * (1 - i) + AtoB * i) * mix + (BtoA * (1 - i) + B * i) * (1 - mix)

        style_layers[l + 1] = style_target
    return style_layers


def random_rotation(N):
    """
    Draws random N-dimensional rotation matrix (det = 1, inverse = transpose) from the special orthogonal group

    From https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/stats/_multivariate.py#L3309
    """
    H = torch.eye(N, device=device)
    D = torch.empty((N,), device=device)
    for n in range(N - 1):
        x = torch.randn(N - n, device=device)
        norm2 = x @ x
        x0 = x[0].item()
        D[n] = torch.sign(x[0]) if x[0] != 0 else 1
        x[0] += D[n] * torch.sqrt(norm2)
        x /= torch.sqrt((norm2 - x0 ** 2 + x[0] ** 2) / 2.0)
        H[:, n:] -= torch.outer(H[:, n:] @ x, x)
    D[-1] = (-1) ** (N - 1) * D[:-1].prod()
    H = (D * H.T).T
    return H


def get_iters_and_sizes(size, iters, passes, use_multires):
    # more iterations for smaller sizes and deeper layers

    if use_multires:
        iters_per_pass = np.arange(2 * passes, passes, -1)
        iters_per_pass = iters_per_pass / np.sum(iters_per_pass) * iters

        sizes = np.linspace(256, size, passes)
        # round to nearest multiple of 32, so that even after 4 max pools the resolution is an even number
        sizes = [int(size + 32 - 1) & -32 for size in sizes]
    else:
        iters_per_pass = np.ones(passes) * int(iters / passes)
        sizes = [size] * passes

    proportion_per_layer = np.array([64, 128, 256, 512, 512]) + 64
    proportion_per_layer = proportion_per_layer / np.sum(proportion_per_layer)
    iters = (iters_per_pass[:, None] * proportion_per_layer[None, :]).astype(np.int32)

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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--style", type=str, nargs="+", action=required_length(1, 2), default="style/graffiti.jpg"
    )
    parser.add_argument("-c", "--content", type=str, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--content_strength", type=float, default=0.01)
    parser.add_argument("--mixing_alpha", type=float, default=0.5)
    parser.add_argument("--hist_mode", type=str, choices=["sym", "pca", "chol", "cdf"], default="chol")
    parser.add_argument("--no_pca", action="store_true")
    parser.add_argument("--no_multires", action="store_true")
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    output = optimal_texture(**vars(args))

    util.save_image(output, args)
