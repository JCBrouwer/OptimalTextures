import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from histmatch import *
from style import *

if __name__ == "__main__":
    # rotations
    mat = torch.randn(8, 8, device=dev)
    rot = random_rotation(8)
    assert rot.det().allclose(torch.tensor(1.0))
    assert (mat @ rot @ rot.T).allclose(mat, rtol=1e-4)  # relative tolerance 10x higher than default

    # histogram matching
    contim = np.asarray(Image.open("content/-166.jpg"))
    content = (torch.from_numpy(contim).permute(2, 0, 1)[None, ...].float() / 255).to(device)
    stylim = np.asarray(Image.open("style/candy.jpg").resize((contim.shape[1], contim.shape[0])))
    style = (torch.from_numpy(stylim).permute(2, 0, 1)[None, ...].float() / 255).to(device)

    num_repeats = 100

    t = time.time()
    for _ in range(num_repeats):
        matched = (pca_match(content, style) * 255)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    print("pca", (time.time() - t) / num_repeats)
    Image.fromarray(np.concatenate((contim, stylim, matched), axis=1)).save("output/pca_match.png")
    fig, ax = plt.subplots(1, 3)
    ax[0].hist(contim.reshape(3, -1).sum(0), bins=128)
    ax[1].hist(stylim.reshape(3, -1).sum(0), bins=128)
    ax[2].hist(matched.reshape(3, -1).sum(0), bins=128)
    plt.show()

    t = time.time()
    for _ in range(num_repeats):
        matched = (cdf_match(content, style) * 255)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    print("cdf", (time.time() - t) / num_repeats)
    Image.fromarray(np.concatenate((contim, stylim, matched), axis=1)).save("output/cdf_match.png")
    fig, ax = plt.subplots(1, 3)
    ax[0].hist(contim.reshape(3, -1).sum(0), bins=128)
    ax[1].hist(stylim.reshape(3, -1).sum(0), bins=128)
    ax[2].hist(matched.reshape(3, -1).sum(0), bins=128)
    plt.show()

    # color channel transfer
    matched = swap_color_channel(
        Image.open("content/-166.jpg"),
        Image.open("style/candy.jpg"),
    ).save("output/color_swap.png")
