import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from vgg import Decoder, Encoder
from histmatch import *

torch.set_grad_enabled(False)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_rotation(N):
    """
    Draws random N-dimensional rotation matrix (det = 1, inverse = transpose)

    From https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/stats/_multivariate.py#L3309
    """
    H = torch.eye(N, device=dev)
    D = torch.empty((N,), device=dev)
    for n in range(N - 1):
        x = torch.randn(N - n, device=dev)
        norm2 = torch.dot(x, x)
        x0 = x[0].item()
        D[n] = torch.sign(x[0]) if x[0] != 0 else 1
        x[0] += D[n] * torch.sqrt(norm2)
        x /= torch.sqrt((norm2 - x0 ** 2 + x[0] ** 2) / 2.0)
        H[:, n:] -= torch.outer(H[:, n:] @ x, x)
    D[-1] = (-1) ** (N - 1) * D[:-1].prod()
    H = (D * H.T).T
    return H


def rotate(tens, rot):
    return (tens.view(tens.size(1), -1).T @ rot).view(tens.shape)


if __name__ == "__main__":
    style = (torch.from_numpy(np.asarray(Image.open(sys.argv[1])).copy()).permute(2, 0, 1)[None, ...].float() / 255).to(
        dev
    )
    output = torch.randn(style.shape, device=dev)

    # multiple resolutions (e.g. each pass can be done for a new resolution)
    num_passes = 5
    pbar = tqdm(total=64 + 128 + 256 + 512 + 512, smoothing=1)
    for _ in range(num_passes):
        # PCA goes here
        for layer in range(1, 6):
            with Encoder(layer).to(dev) as encoder:
                style_layer = encoder(style)
                output_layer = encoder(output)

            N = output_layer.shape[1]  # channels
            for i in range(int(N / num_passes)):
                rotation = random_rotation(N)
                rotated_style = rotate(style_layer, rotation)
                rotated_output = rotate(output_layer, rotation)

                matched_output = match_histogram(rotated_output, rotated_style)

                output_layer = rotate(matched_output, rotation.T)

                pbar.update(1)

            with Decoder(layer).to(dev) as decoder:
                output = decoder(output_layer)

    Image.fromarray((output * 255)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)).save("output/texture.png")