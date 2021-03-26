# OptimalTextures
An implementation of [Optimal Textures: Fast and Robust Texture Synthesis and Style Transfer through Optimal Transport](https://arxiv.org/abs/2010.14702) for TU Delft CS4240.

This repository is a WIP.

[Paper notes](notes.md)

```bash
pip install -r requirements
python style.py
```

```
usage: style.py [-h] [-s STYLE] [-c CONTENT] [--size SIZE]
                [--content_strength CONTENT_STRENGTH]
                [--hist_mode {sym,pca,chol}] [--no_pca] [--no_multires]
                [--covariance] [--num_passes NUM_PASSES]

optional arguments:
  -h, --help            show this help message and exit
  -s STYLE, --style STYLE
  -c CONTENT, --content CONTENT
  --size SIZE
  --content_strength CONTENT_STRENGTH
  --hist_mode {sym,pca,chol}
  --no_pca
  --no_multires
  --covariance
  --num_passes NUM_PASSES
```