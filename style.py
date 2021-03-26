import argparse
import sys

import torch
import torchvision
from tqdm import tqdm

import util
from histmatch import *
from util import name
from vgg import Decoder, Encoder

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def spatial(tensor, shape):
    return tensor.T.reshape(1, tensor.shape[1], *shape[:2])


def flatten(tensor):
    return tensor.squeeze().permute(1, 2, 0).reshape(-1, tensor.shape[1])


def encode(tensor, layer, eigvecs):
    with Encoder(layer).to(device) as encoder:
        features = encoder(tensor).squeeze().permute(1, 2, 0)  # remove batch channel and move channels to last axis
        spatial_shape = features.shape
        features = features.reshape(-1, features.shape[2])  # [pixels, channels]

        if not args.no_pca:
            if eigvecs is None:
                if args.covariance:
                    A = (features - features.mean()).T @ (features - features.mean()) / features.shape[1]
                else:
                    A = features - features.mean()
                _, eigvals, eigvecs = torch.svd(A)
                total = torch.sum(eigvals)
                k = np.argmax(np.cumsum([(i / total) for i in eigvals]) > 0.9)
                eigvecs = eigvecs[:, :k]  # the vectors for 90% of variance will be kept

            features = features @ eigvecs

        return features, eigvecs, spatial_shape


def encode_inputs(style, content):
    style_layers, style_shapes, style_eigvs, content_layers = [None], [None], [None], [None]
    for layer in range(1, 6):
        style_layer, eigvecs, style_shape = encode(style, layer, None)

        style_layers.append(style_layer)
        style_shapes.append(style_shape)
        style_eigvs.append(eigvecs)

        if content is not None:
            content_layer, _, _ = encode(content, layer, eigvecs)
            content_layer -= content_layer.mean()
            content_layer += style_layer.mean()
            content_layers.append(content_layer)

    return style_layers, style_shapes, style_eigvs, content_layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--style", type=str, default="style/graffiti.jpg")
    parser.add_argument("-c", "--content", type=str, default=None)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--content_strength", type=float, default=0.125)
    parser.add_argument("--hist_mode", type=str, choices=["sym", "pca", "chol"], default="chol")
    parser.add_argument("--no_pca", action="store_true")
    parser.add_argument("--no_multires", action="store_true")
    parser.add_argument("--covariance", action="store_true")
    parser.add_argument("--num_passes", type=int, default=5)
    args = parser.parse_args()

    sizes = [args.size]
    if not args.no_multires:
        sizes = np.linspace(256, args.size, args.num_passes)
        sizes = [int(size + 32 - 1) & -32 for size in sizes]
        # round to nearest multiple of 32, so that even after 4 max pools the resolution is an even number

    style = util.load_image(args.style, size=sizes[0])

    content = None
    if args.content is not None:
        content = util.load_image(args.content, size=sizes[0])
        content_strength = float(args.content_strength)

    output = torch.rand((1, 3, sizes[0], sizes[0]), device=device)

    style_layers, style_shapes, style_eigvs, content_layers = encode_inputs(style, content)

    pbar = tqdm(total=64 + 128 + 256 + 512 + 512, smoothing=1)
    for i in range(args.num_passes):

        if i != 0 and not args.no_multires:
            output = torch.nn.functional.interpolate(
                output, size=(sizes[i], sizes[i]), mode="bicubic", align_corners=False
            )

            style = util.load_image(args.style, size=sizes[i])
            if content is not None:
                content = util.load_image(args.content, size=sizes[i])

            style_layers, style_shapes, style_eigvs, content_layers = encode_inputs(style, content)

        for layer in range(5, 0, -1):
            output_layer, _, shape = encode(output, layer, style_eigvs[layer])

            for it in range(int(shape[-1] / args.num_passes)):
                rotation = random_rotation(output_layer.shape[1])

                proj_s = style_layers[layer] @ rotation
                proj_o = output_layer @ rotation

                match_o = flatten(
                    hist_match(
                        spatial(proj_o, shape),
                        spatial(proj_s, style_shapes[layer]),
                        mode=args.hist_mode,
                    )
                )
                output_layer = match_o @ rotation.T

                if content is not None and layer >= 3:
                    strength = args.content_strength
                    if layer == 4:
                        strength /= 2
                    elif layer == 3:
                        strength /= 4
                    output_layer += strength * (content_layers[layer] - output_layer)

                pbar.update(1)

            if not args.no_pca:
                output_layer = output_layer @ style_eigvs[layer].T

            with Decoder(layer).to(device) as decoder:
                output = decoder(spatial(output_layer, shape))

    outname = name(args.style)
    if content is not None:
        outname += name(args.content) + "_" + str(args.content_strength)
    outname += "_" + args.hist_mode
    if not args.no_pca:
        outname += "_pca"
    if not args.no_multires:
        outname += "_multires"
    torchvision.utils.save_image(output, f"output/{outname}.png")
