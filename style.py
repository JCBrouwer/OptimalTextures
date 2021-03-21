import sys

import numpy as np
import torch
from PIL import Image

import sliced_hist
from encoder_decoder_factory import Decoder, Encoder

with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoders = [Encoder(i).to(device) for i in range(1, 6)]
    decoders = [Decoder(i).to(device) for i in range(1, 6)]

    style = (
        torch.from_numpy(np.asarray(Image.open("../style/graffiti256.jpg"))).permute(2, 0, 1)[None, ...].float() / 255
    ).to(device)

    output = torch.randn(style.shape, device=device)

    # multiple resolutions (e.g. each pass can be done for a new resolution)
    num_passes = 5
    for _ in range(num_passes):
        # PCA goes here
        for layer in range(5):
            style_layer = encoders[layer](style)
            output_layer = encoders[layer](output)

            output_layer = sliced_hist.optimal_transport(output_layer, style_layer, passes=num_passes)

            output = decoders[layer](output_layer)

    Image.fromarray((output * 255)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)).save("texture.png")
