# OptimalTextures
An implementation of [Optimal Textures: Fast and Robust Texture Synthesis and Style Transfer through Optimal Transport](https://arxiv.org/abs/2010.14702) for TU Delft CS4240.

This repository is a WIP.

[Paper notes](notes.md)

```bash
git clone https://github.com/JCBrouwer/OptimalTextures
cd OptimalTextures
pip install -r requirements
python optex.py -h
python optex.py -s style/graffiti-small.jpg --size 512  # texture synthesis
python optex.py -s style/lava-small.jpg -c content/rocket.jpg --content_strength 0.01  # style transfer
python optex.py -s style/zebra.jpg style/pattern-small.jpg --mixing_alpha 0.5  # texture mixing
```