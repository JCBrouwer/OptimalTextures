import os
import random

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from histmatch import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def swap_color_channel(target, source, colorspace="HSV"):  # YCbCr also works
    target_channels = list(target.convert(colorspace).split())
    source_channels = list(source.resize(target.size).convert(colorspace).split())
    target_channels[0] = source_channels[0]
    return Image.merge(colorspace, target_channels).convert("RGB")


def match_histogram(target, source, strategy="cdf"):
    if strategy == "pca":
        return pca_match(target, source)
    else:
        return cdf_match(target, source)


def random_rotation(N):
    """
    Draws random N-dimensional rotation matrix (det = 1, inverse = transpose)

    From https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/stats/_multivariate.py#L3309
    """
    H = torch.eye(N, device=device)
    D = torch.empty((N,), device=device)
    for n in range(N - 1):
        x = torch.randn(N - n, device=device)
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


def optimal_transport(output, style, passes):
    N = output.shape[1]  # channels
    for _ in tqdm(range(int(N / passes))):
        rotation = random_rotation(N)
        rotated_style = rotate(style, rotation)
        rotated_output = rotate(output, rotation)
        matched_output = match_histogram(rotated_output, rotated_style)
        output = rotate(matched_output, rotation.T)
    return output


if __name__ == "__main__":
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    mat = torch.randn(8, 8)
    rot = random_rotation(8)
    assert rot.det() == 1
    assert (mat @ rot @ rot.T).allclose(mat, rtol=5e-5)  # relative tolerance 5x higher than default

    import matplotlib.pyplot as plt

    contim = np.asarray(Image.open("../content/-166.jpg"))
    content = torch.from_numpy(contim).permute(2, 0, 1)[None, ...].float() / 255
    stylim = np.asarray(Image.open("../style/candy.jpg").resize((contim.shape[1], contim.shape[0])))
    style = torch.from_numpy(stylim).permute(2, 0, 1)[None, ...].float() / 255

    matched = (pca_match(content, style) * 255)[0].permute(1, 2, 0).numpy().astype(np.uint8)
    Image.fromarray(np.concatenate((contim, stylim, matched), axis=1)).save("pca_match.png")

    fig, ax = plt.subplots(1, 3)
    ax[0].hist(contim.reshape(3, -1).sum(0), bins=128)
    ax[1].hist(stylim.reshape(3, -1).sum(0), bins=128)
    ax[2].hist(matched.reshape(3, -1).sum(0), bins=128)
    plt.show()

    matched = (cdf_match(content, style, False) * 255)[0].permute(1, 2, 0).numpy().astype(np.uint8)
    Image.fromarray(np.concatenate((contim, stylim, matched), axis=1)).save("cdf_match1.png")

    fig, ax = plt.subplots(1, 3)
    ax[0].hist(contim.reshape(3, -1).sum(0), bins=128)
    ax[1].hist(stylim.reshape(3, -1).sum(0), bins=128)
    ax[2].hist(matched.reshape(3, -1).sum(0), bins=128)
    plt.show()

    matched = (cdf_match(content, style, True) * 255)[0].permute(1, 2, 0).numpy().astype(np.uint8)
    Image.fromarray(np.concatenate((contim, stylim, matched), axis=1)).save("cdf_match2.png")

    fig, ax = plt.subplots(1, 3)
    ax[0].hist(contim.reshape(3, -1).sum(0), bins=128)
    ax[1].hist(stylim.reshape(3, -1).sum(0), bins=128)
    ax[2].hist(matched.reshape(3, -1).sum(0), bins=128)
    plt.show()

    contim = Image.open("../content/-166.jpg")
    stylim = Image.open("../style/candy.jpg")
    matched = swap_color_channel(contim, stylim).save("color_swap.png")
