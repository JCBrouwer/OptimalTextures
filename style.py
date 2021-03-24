import sys

import torch
import torchvision
from PIL import Image
from torchinterp1d import Interp1d
from tqdm import tqdm

import util
from histmatch import *
from vgg import Decoder, Encoder

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
interp = Interp1d()


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


import matplotlib.pyplot as plt

if __name__ == "__main__":
    style = util.load_image(sys.argv[1])
    output = util.load_image(sys.argv[2])
    # output = torch.rand(style.shape, device=device)

    style_layers = [None]  # add one to index so it works better in next loop
    for layer in range(1, 6):
        with Encoder(layer).to(device) as encoder:
            enc_s = encoder(style).squeeze().permute(1, 2, 0)  # remove batch channel and move channels to last axis
            style_layers.append(enc_s.reshape(-1, enc_s.shape[2]))  # [pixels, channels]

    # multiple resolutions (e.g. each pass can be done for a new resolution ?)
    num_passes = 1
    pbar = tqdm(total=64 + 128 + 256 + 512 + 512, smoothing=1)
    for _ in range(num_passes):
        # PCA goes here
        for layer in range(1, 6):
            with Encoder(layer).to(device) as encoder:
                output_layer = encoder(output).squeeze().permute(1, 2, 0)
                h, w, c = output_layer.shape
                output_layer = output_layer.reshape(-1, c)  # [pixels, channels]

            for it in range(int(c / num_passes)):
                rotation = random_rotation(c)

                # print(output_layer.min(), output_layer.mean(), output_layer.max())
                # print(style_layers[layer].min(), style_layers[layer].mean(), style_layers[layer].max())

                proj_s = style_layers[layer] @ rotation
                # print(proj_s.min(), proj_s.mean(), proj_s.max())
                proj_o = output_layer @ rotation
                # print(proj_o.min(), proj_o.mean(), proj_o.max())

                match_o = cdf_match(proj_o, proj_s)
                # print(match_o.min(), match_o.mean(), match_o.max())

                output_layer = match_o @ rotation.T
                # print(output_layer.min(), output_layer.mean(), output_layer.max())

                # with Decoder(layer).to(device) as decoder:
                #     plt.figure()
                #     out = decoder(output_layer.T.reshape(1, c, h, w).clamp(0, output_layer.max()))
                #     # print(out.min(), out.mean(), out.max())
                #     plt.imshow(out.cpu().numpy().squeeze().transpose(1, 2, 0))
                #     plt.show(block=False)

                # print()
                pbar.update(1)

            with Decoder(layer).to(device) as decoder:
                output = decoder(output_layer.T.reshape(1, c, h, w))

    torchvision.utils.save_image(torch.cat((style, output)), "output/texture.png")
    plt.show()
