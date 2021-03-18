import sys

import numpy as np
import torch
from PIL import Image

import sliced_hist

sys.path.append("wct")
from wct.encoder_decoder_factory import Decoder, Encoder

encoders = [Encoder(i) for i in range(1, 6)]
decoders = [Decoder(i) for i in range(1, 6)]

style = torch.from_numpy(np.asarray(Image.open("../style/graffiti.jpg"))).permute(2, 0, 1)[None, ...].float() / 255

output = torch.randn(style.shape)

num_passes = 5
for _ in range(num_passes):
    for layer in range(5):
        style_layer = encoders[layer](style)
        output_layer = encoders[layer](output)
        output_layer = sliced_hist.optimal_transport(output_layer, style_layer, passes=num_passes)
        output = decoders[layer](output_layer)

Image.fromarray((output * 255)[0].permute(1, 2, 0).numpy().astype(np.uint8)).save("texture.png")
