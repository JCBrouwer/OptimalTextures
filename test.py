import time

import matplotlib.pyplot as plt
import torch
from PIL import Image

from histmatch import *
from style import *
from vgg import Decoder, Encoder

if __name__ == "__main__":

    # histogram matching

    def plot_hists(imgs):
        _, ax = plt.subplots(2, 3, figsize=(16, 9))
        for i, img in enumerate(imgs):
            ax[0, i].hist(img[:, :, 0].cpu().numpy().ravel(), bins=128, color="r", alpha=0.333)
            ax[0, i].hist(img[:, :, 1].cpu().numpy().ravel(), bins=128, color="g", alpha=0.333)
            ax[0, i].hist(img[:, :, 2].cpu().numpy().ravel(), bins=128, color="b", alpha=0.333)
            ax[1, i].imshow(img.squeeze().permute(1, 2, 0).cpu().numpy())
        plt.tight_layout()
        plt.show(block=False)

    content = util.load_image("content/rocket.jpg")
    style = util.load_image("style/candy.jpg")

    num_repeats = 100

    t = time.time()
    for _ in range(num_repeats):
        matched = hist_match(content, style, mode="chol").reshape(content.shape)
    print("chol", (time.time() - t) / num_repeats)
    plot_hists((content, style, matched))

    t = time.time()
    for _ in range(num_repeats):
        matched = hist_match(content, style, mode="pca").reshape(content.shape)
    print("pca", (time.time() - t) / num_repeats)
    plot_hists((content, style, matched))

    t = time.time()
    for _ in range(num_repeats):
        matched = hist_match(content, style, mode="sym").reshape(content.shape)
    print("sym", (time.time() - t) / num_repeats)
    plot_hists((content, style, matched))

    t = time.time()
    for _ in range(num_repeats):
        matched = hist_match(content, style, mode="cdf").reshape(content.shape)
    print("cdf", (time.time() - t) / num_repeats)
    plot_hists((content, style, matched))

    # rotations
    mat = torch.randn(16, 16, device=device)
    rot = random_rotation(16)
    assert rot.det().allclose(torch.tensor(1.0))
    assert (rot.T).allclose(torch.inverse(rot), rtol=5e-3)  # relative tolerance 50x higher than default

    # encode decode uniform noise
    fig, ax = plt.subplots(2, 5, figsize=(16, 9))
    og_style = util.load_image("style/graffiti256.jpg")
    output = torch.rand_like(og_style)
    ax[0, 2].imshow(output.clamp(0, 1).cpu().detach().squeeze().permute(1, 2, 0).numpy())
    for layer in range(5, 0, -1):
        with Encoder(layer).to(device) as encoder, Decoder(layer).to(device) as decoder:
            style_layer = encoder(og_style)
            b, c, h, w = style_layer.shape
            style_layer = style_layer.reshape(-1, c)
            output_layer = encoder(output).reshape(-1, c)
            output = decoder(output_layer.reshape(b, c, h, w))
            ax[1, layer - 1].imshow(output.clamp(0, 1).squeeze().permute(1, 2, 0).cpu().numpy())
    [fig.delaxes(ax.flatten()[i]) for i in [0, 1, 3, 4]]
    plt.tight_layout()
    plt.suptitle("Encode and decode noise")
    plt.show(block=False)

    # encode decode style
    fig, ax = plt.subplots(2, 5, figsize=(16, 9))
    og_style = util.load_image("style/graffiti256.jpg")
    ax[0, 2].imshow(og_style.clamp(0, 1).squeeze().permute(1, 2, 0).cpu().numpy())
    for layer in range(5, 0, -1):
        with Encoder(layer).to(device) as encoder, Decoder(layer).to(device) as decoder:
            style_layer = encoder(og_style).squeeze().permute(1, 2, 0)
            h, w, c = style_layer.shape
            style_layer = style_layer.reshape(-1, c)
            style = decoder(style_layer.T.reshape(1, c, h, w))
            ax[1, layer - 1].imshow(style.clamp(0, 1).squeeze().permute(1, 2, 0).cpu().numpy())
    [fig.delaxes(ax.flatten()[i]) for i in [0, 1, 3, 4]]
    plt.tight_layout()
    plt.suptitle("Encode and decode style")
    plt.show(block=False)

    # encode decode transport
    fig, ax = plt.subplots(2, 5, figsize=(16, 9))
    og_style = util.load_image("style/graffiti256.jpg")
    output = torch.rand_like(og_style)
    ax[0, 2].imshow(output.clamp(0, 1).cpu().squeeze().permute(1, 2, 0).numpy())
    for layer in range(5, 0, -1):
        with Encoder(layer).to(device) as encoder, Decoder(layer).to(device) as decoder:
            style_layer = encoder(og_style).squeeze().permute(1, 2, 0)
            style_layer = style_layer.reshape(-1, style_layer.shape[2])
            output_layer = encoder(output).squeeze().permute(1, 2, 0)
            h, w, c = output_layer.shape
            output_layer = output_layer.reshape(-1, c)
            for _ in range(10):
                rotation = random_rotation(c)
                proj_s = style_layer @ rotation
                proj_o = output_layer @ rotation
                match_o = flatten(
                    hist_match(
                        spatial(proj_o, (h, w, c)),
                        spatial(proj_s, (h, w, c)),
                        mode=args.hist_mode,
                    )
                )
                output_layer = match_o @ rotation.T
            output = decoder(output_layer.T.reshape(1, c, h, w))
            ax[1, layer - 1].imshow(output.clamp(0, 1).squeeze().permute(1, 2, 0).cpu().numpy())
    [fig.delaxes(ax.flatten()[i]) for i in [0, 1, 3, 4]]
    plt.tight_layout()
    plt.suptitle("Encode, three iters of hist match, and decode")
    plt.show(block=False)

    # color channel transfer
    plt.figure()
    plt.imshow(swap_color_channel(Image.open("content/rocket.jpg"), Image.open("style/graffiti256.jpg")))
    plt.show()
