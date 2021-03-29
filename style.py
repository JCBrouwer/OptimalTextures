import argparse

import torch
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


def encode(tensor, l, eigvecs):
    with Encoder(l).to(device) as encoder:
        features = encoder(tensor).permute(0, 2, 3, 1)  # [b, h, w, c]

        if not args.no_pca:
            if eigvecs is None:
                flat_feat = features.reshape(-1, features.shape[-1])
                if args.covariance:
                    A = (flat_feat - flat_feat.mean()).T @ (flat_feat - flat_feat.mean()) / flat_feat.shape[1]
                else:
                    A = flat_feat - flat_feat.mean()
                _, eigvals, eigvecs = torch.svd(A)
                total = torch.sum(eigvals)
                k = np.argmax(np.cumsum([(i / total) for i in eigvals]) > 0.9)
                eigvecs = eigvecs[:, :k]  # the vectors for 90% of variance will be kept

            features = features @ eigvecs

        return features, eigvecs


def encode_inputs(style, content):
    style_layers, style_eigvs, content_layers = [None], [None], [None]
    for l in range(1, 6):
        style_layer, eigvecs = encode(style, l, eigvecs=None)

        style_layers.append(style_layer)
        style_eigvs.append(eigvecs)

        if content is not None:
            content_layer, _ = encode(content, l, eigvecs)
            content_layer -= content_layer.mean()
            content_layer += style_layer.mean()
            content_layers.append(content_layer)

    return style_layers, style_eigvs, content_layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--style", type=str, default="style/graffiti.jpg")
    parser.add_argument("-c", "--content", type=str, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--content_strength", type=float, default=0.005)
    parser.add_argument("--hist_mode", type=str, choices=["sym", "pca", "chol", "cdf"], default="chol")
    parser.add_argument("--no_pca", action="store_true")
    parser.add_argument("--no_multires", action="store_true")
    parser.add_argument("--covariance", action="store_true")
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--iters", type=int, default=1500)
    args = parser.parse_args()

    iters_per_pass = np.ones(args.passes) * int(args.iters / args.passes)
    proportion_per_layer = np.array([64, 128, 256, 512, 512])
    proportion_per_layer = proportion_per_layer / np.sum(proportion_per_layer)

    sizes = [args.size]
    if not args.no_multires:
        sizes = np.linspace(256, args.size, args.passes)
        sizes = [int(size + 32 - 1) & -32 for size in sizes]
        # round to nearest multiple of 32, so that even after 4 max pools the resolution is an even number

        iters_per_pass = np.arange(2 * args.passes, args.passes, -1)
        iters_per_pass = iters_per_pass / np.sum(iters_per_pass) * args.iters

    iters = (iters_per_pass[:, None] * proportion_per_layer[None, :]).astype(np.int32)

    style = util.load_image(args.style, size=sizes[0])

    content = None
    if args.content is not None:
        content = util.load_image(args.content, size=sizes[0])
        content_strength = float(args.content_strength)

    output = torch.rand((1, 3, sizes[0], sizes[0]), device=device)

    style_layers, style_eigvs, content_layers = encode_inputs(style, content)

    pbar = tqdm(total=args.iters, smoothing=1)
    for p in range(args.passes):

        if p != 0 and not args.no_multires:
            output = torch.nn.functional.interpolate(
                output, size=(sizes[p], sizes[p]), mode="bicubic", align_corners=False
            )

            style = util.load_image(args.style, size=sizes[p])
            if content is not None:
                content = util.load_image(args.content, size=sizes[p])

            style_layers, style_eigvs, content_layers = encode_inputs(style, content)

        for l in range(5, 0, -1):
            output_layer, _ = encode(output, l, style_eigvs[l])

            for _ in range(iters[p, l - 1]):
                rotation = random_rotation(output_layer.shape[-1])

                proj_s = style_layers[l] @ rotation
                proj_o = output_layer @ rotation

                match_o = hist_match(proj_o, proj_s, mode=args.hist_mode)

                output_layer = match_o @ rotation.T

                if content is not None and l >= 3:
                    strength = args.content_strength
                    if l == 4:
                        strength /= 2
                    elif l == 3:
                        strength /= 4
                    output_layer += strength * (content_layers[l] - output_layer)

                pbar.update(1)

            if not args.no_pca:
                output_layer = output_layer @ style_eigvs[l].T

            with Decoder(l).to(device) as decoder:
                output = decoder(output_layer.permute(0, 3, 1, 2))

    util.save_image(output, args)
