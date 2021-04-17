import argparse
from functools import partial

import torch
from torch.nn.functional import interpolate
from tqdm import tqdm

import util
from histmatch import *
from vgg import Decoder, Encoder
from kornia.color.hls import rgb_to_hls, hls_to_rgb

downsample = partial(interpolate, mode="nearest")  # for mixing mask
upsample = partial(interpolate, mode="bicubic", align_corners=False)  # for output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def optimal_texture(
    style,
    content=None,
    size=512,
    content_strength=0.1,
    mixing_alpha=0.5,
    style_scale=1,
    oversize_style=False,
    hist_mode="pca",
    color_transfer=None,
    no_pca=False,
    no_multires=False,
    passes=5,
    iters=500,
    seed=None,
    **kwargs,
):
    if seed is not None:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # readability booleans
    use_pca = not no_pca
    use_multires = not no_multires
    texture_mixing = len(style) > 1

    # get number of iterations and sizes for optization
    iters_per_pass_and_layer, sizes = get_iters_and_sizes(size, iters, passes, use_multires)

    # load inputs and initialize output image
    styles = util.load_styles(style, size=sizes[0], scale=style_scale, oversize=oversize_style)
    content = util.maybe_load_content(content, size=sizes[0])
    output = torch.rand(content.shape if content is not None else (1, 3, sizes[0], sizes[0]), device=device)

    # transform style and content to VGG feature space
    style_layers, style_eigvs, content_layers = encode_inputs(styles, content, use_pca=use_pca)

    if texture_mixing:
        assert styles[0].shape == styles[1].shape, "Texture mixing requires both styles to have the same dimensions"
        mixing_mask = torch.ceil(torch.rand(style_layers[1].shape[1:3], device=device) - mixing_alpha)[None, None, ...]
        style_layers = mix_style_layers(style_layers, mixing_mask, mixing_alpha, hist_mode)

    pbar = tqdm(total=iters, smoothing=0)
    for p in range(passes):

        if use_multires and p != 0:
            # reload style and content at the new size
            styles = util.load_styles(args.style, size=sizes[p], scale=style_scale, oversize=oversize_style)
            content = util.maybe_load_content(args.content, size=sizes[p])

            # upsample to next size
            output = upsample(output, size=content.shape[2:] if content is not None else (sizes[p], sizes[p]))

            # get style and content target features
            style_layers, style_eigvs, content_layers = encode_inputs(styles, content, use_pca=use_pca)

            if texture_mixing:
                style_layers = mix_style_layers(style_layers, mixing_mask, mixing_alpha, hist_mode)

        for l in range(5, 0, -1):
            pbar.set_description(f"Current resolution: {sizes[p if use_multires else 0]}, current layer relu{l}_1")

            # encode layer to VGG feature space
            with Encoder(l).to(device) as encoder:
                output_layer = encoder(output)
            if use_pca:
                output_layer = output_layer @ style_eigvs[l]  # project onto principal components

            for _ in range(iters_per_pass_and_layer[p, l - 1]):
                output_layer = optimal_transport(output_layer, style_layers[l], hist_mode)

                # apply content matching step
                if content is not None and l >= 3:
                    strength = content_strength
                    strength /= 2 ** (5 - l)  # 1, 2, or 4 depending on layer depth
                    output_layer += strength * (content_layers[l] - output_layer)

                pbar.update(1)

            if use_pca:
                output_layer = output_layer @ style_eigvs[l].T  # reverse principal component projection
            with Decoder(l).to(device) as decoder:
                output = decoder(output_layer)  # decode back to image space

    if color_transfer is not None:
        target_hls = rgb_to_hls(content)
        target_hls[:, 1] = rgb_to_hls(output)[:, 1]  # swap lightness channel
        target = hls_to_rgb(target_hls)

        if color_transfer == "opt":
            target, output = target.permute(0, 2, 3, 1), output.permute(0, 2, 3, 1)  # [b, h, w, c]
            for _ in range(3):
                output = optimal_transport(output, target, "cdf")
            output = output.permute(0, 3, 1, 2)  # [b, c, h, w]

        elif color_transfer == "lum":
            return target  # return output with hue and saturation from content

    return output


def optimal_transport(output_layer, style_layer, hist_mode):
    rotation = random_rotation(output_layer.shape[-1])

    rotated_output = output_layer @ rotation
    rotated_style = style_layer @ rotation

    matched_output = hist_match(rotated_output, rotated_style, mode=hist_mode)

    output_layer = matched_output @ rotation.T  # rotate back to normal

    return output_layer


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
    k = (torch.cumsum(eigvals / torch.sum(eigvals), dim=0) > 0.9).max(0).indices.item()
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
        sizes = [util.round32(size) for size in sizes]
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
        "-s",
        "--style",
        type=str,
        nargs="+",
        action=required_length(1, 2),
        default=["style/graffiti.jpg"],
        help="Example(s) of the style your texture should take",
    )
    parser.add_argument(
        "-c", "--content", type=str, default=None, help="The structure/shape you want your image to take"
    )
    parser.add_argument(
        "--size", type=int, default=512, help="The output size of the image (larger output = more memory/time required)"
    )
    parser.add_argument(
        "--style_scale",
        type=float,
        default=1,
        help="Scale the style relative to the generated image. Will affect the scale of details generated.",
    )
    parser.add_argument(
        "--oversize_style",
        action="store_true",
        help="Allow scaling of style larger than its original size. Might cause blurry outputs.",
    )
    parser.add_argument(
        "--content_strength",
        type=float,
        default=0.01,
        help="Strength with which to focus on the structure in your content image.",
    )
    parser.add_argument(
        "--mixing_alpha", type=float, default=0.5, help="Value between 0 and 1 for interpolation between 2 textures"
    )
    parser.add_argument(
        "--hist_mode",
        type=str,
        choices=["sym", "pca", "chol", "cdf"],
        default="pca",
        help="Histogram matching strategy. CDF is slower than the others, but may use less memory. Each gives slightly different results.",
    )
    parser.add_argument(
        "--color_transfer",
        type=str,
        default=None,
        choices=["lum", "opt"],
        help="Strategy to employ to keep original color of content image.",
    )
    parser.add_argument("--no_pca", action="store_true", help="Disable PCA of features (slower).")
    parser.add_argument(
        "--no_multires",
        action="store_true",
        help="Disable multi-scale rendering (slower, less long-range texture qualities).",
    )
    parser.add_argument(
        "--passes", type=int, default=5, help="Number of times to loop over each of the 5 layers in VGG-19"
    )
    parser.add_argument("--iters", type=int, default=500, help="Total number of iterations to optimize.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random number generator.")
    parser.add_argument("--output_dir", type=str, default="output/", help="Directory to output results.")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    output = optimal_texture(**vars(args))

    util.save_image(output, args)
