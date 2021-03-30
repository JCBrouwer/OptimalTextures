# OptimalTextures
An implementation of [Optimal Textures: Fast and Robust Texture Synthesis and Style Transfer through Optimal Transport](https://arxiv.org/abs/2010.14702) for TU Delft CS4240.

This repository is a WIP.

[Paper notes](notes.md)

```bash
pip install -r requirements
python style.py -h
python style.py -s style/graffiti-small.jpg  # texture synthesis
python style.py -s style/lava-small.jpg -c content/rocket.jpg   # style transfer
python style.py -s style/zebra.jpg style/pattern-small.jpg   # texture mixing
```