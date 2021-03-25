import sys

import torch
import torchvision
from tqdm import tqdm

import util
from histmatch import *
from vgg import Decoder, Encoder

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


if __name__ == "__main__":
    pca = False

    style = util.load_image(sys.argv[1])
    content = None
    if len(sys.argv) > 2:
        content = util.load_image(sys.argv[2])
        content_strength = float(sys.argv[3])
    output = torch.rand(style.shape, device=device)

    # add one to index so it works better in next loop
    style_layers, style_pcas, content_layers = [None], [None], [None]
    for layer in range(1, 6):
        with Encoder(layer).to(device) as encoder:

            enc_style = encoder(style).squeeze().permute(1, 2, 0)  # remove batch channel and move channels to last axis
            enc_style = enc_style.reshape(-1, enc_style.shape[2])  # [pixels, channels]
            style_layers.append(enc_style)

            if pca:
                enc_style -= enc_style.mean()
                U, S, V = torch.svd(enc_style)
                total = torch.sum(S)
                k = 20  # torch.searchsorted(np.cumsum([(i / total) for i in sorted(S, reverse=True)]), torch.tensor([0.9]))
                eigvecs = U.T[:, :k]  # the first k vectors will be kept
                smaller_style = U @ eigvecs

                style_layers.append(smaller_style)
                style_pcas.append(eigvecs)

            if content is not None:
                enc_content = encoder(content).squeeze().permute(1, 2, 0)
                enc_content = enc_content.reshape(-1, enc_content.shape[2])

                if pca:
                    enc_content = enc_content @ style_pcas[-1]

                enc_content -= enc_content.mean()
                enc_content += style_layers[-1].mean()
                content_layers.append(enc_content)

    num_passes = 5
    pbar = tqdm(total=64 + 128 + 256 + 512 + 512, smoothing=1)
    for i in range(num_passes):

        # if i != 0:
        #     output = torch.nn.functional.interpolate(output, scale_factor=1.5)

        for layer in range(5, 0, -1):
            with Encoder(layer).to(device) as encoder:
                output_layer = encoder(output).squeeze().permute(1, 2, 0)
                h, w, c = output_layer.shape
                output_layer = output_layer.reshape(-1, c)  # [pixels, channels]

                if pca:
                    output_layer = output_layer @ style_pcas[-1]
                    c = output_layer.shape[1]

            for it in range(int(c / num_passes)):
                rotation = random_rotation(c)

                proj_s = style_layers[layer] @ rotation
                proj_o = output_layer @ rotation

                match_o = hist_match_np(proj_o, proj_s)

                output_layer = match_o @ rotation.T

                if content is not None and layer >= 3:
                    strength = content_strength
                    if layer == 4:
                        strength /= 2
                    elif layer == 3:
                        strength /= 4
                    output_layer += strength * (content_layers[layer] - output_layer)

                if pca:
                    output_layer = output_layer @ style_pcas[-1].T

                pbar.update(1)

            with Decoder(layer).to(device) as decoder:
                output = decoder(output_layer.T.reshape(1, c, h, w))

    torchvision.utils.save_image(torch.cat((style, output)), "output/texture.png")
