# Replicating "Optimal Textures: Fast and Robust Texture Synthesis and Style Transfer through Optimal Transport"

Hans Brouwer & Jim Kok

In this post we give an overview of our replication of the paper [Optimal Textures: Fast and Robust Texture Synthesis and Style Transfer through Optimal Transport](https://arxiv.org/abs/2010.14702) for the TU Delft Deep Learning course CS4240. Our implementation in PyTorch can be found [here](https://github.com/JCBrouwer/OptimalTextures).

## Overview
Optimal Textures presents a new approach for texture synthesis and style transfer. The presented algorithm can directly optimize the histograms of intermediate features of a model (VGG19 in this case) to match the statistics of a target image. This avoids the costly backpropagation which is required by other approaches which try to match 2nd order statistics like the Gram matrix (the correlations between intermediate features). Compared to other algorithms which seek to speed up exemplar based image generation, Optimal Textures also achieves a better quality tradeoff.

The approach builds on an autoencoder trained to invert features of a pretrained iamge recognition network (as introduced by Li et al. in [Universal Style Transfer via Feature Transforms](https://arxiv.org/abs/1705.08086)). This autoencoder allows one to encode an image to feature space, perform an optimization on the features, and then decode back to an image.

The core algorithm introduced by Optimal Textures consists of 4 steps:
1. Encode a target and style image to the intermediate features of an image recognition network
2. Apply a random rotation to the features of both images at a given depth in the network
3. Match the histograms of the rotated feature tensors
4. Undo rotation and decode the feature back to image space
<!-- The core algorithm consists of iteratively matching the histograms of intermediate features at different depths within the network. To make this matching more robust, an N-dimensional random rotation is applied before each matching operation (where N is equal to the number of channels in the intermediate features at the given depth). This ensures that the histograms match along any dimension in the N-dimensional space. -->

The rotation used is an N-dimensional ...

<br><div class="image" style="text-align: center; text-color: gray; font-size: 10">
<img src="histmatch.jpg" alt="Matching the histograms of features along a single direction in 2D space." style="max-width:66%">
<div>Matching the histograms of features along a single direction in 2D space. <a href="https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.458.7694&rep=rep1&type=pdf">F. Pitie et al., Automated Colour Grading Using Colour Distribution Transfer</a></div>
</div><br>

This simple algorithm can be augmented by a couple techniques that improve the speed and quality of results. The first technique is to project the matched features to a smaller subspace and perform the optimization there. The chosen subspace is the one spanned by the first principal components that capture 90% of the total variance. This reduces the dimension of the feature tensors while retaining the majority of the expressive capacity of the features. The second technique is to synthesize images starting from a small resolution (256 pixels) and progressively upscaling during optimization to the desired size. This once again reduces the size of the feature tensors (this time along the spatial axes) which improves speed. Another benefit is that longer-range relations in the example image are captured as the receptive field of VGG19's convolutions is relatively larger for the smaller starting images.

The paper further shows how the basic algorithm can be used for multiple tasks similar to texture synthesis: style transfer, color transfer, texture mixing, and mask-guided synthesis.

## Replication

Now we will discuss our implementation of 

### Core Algorithm

### Histogram Matching

### PCA

### Multi-resolution Synthesis

### Style Transfer

### Color Transfer

### Texture Mixing

<!-- 
## Related Work
Since the seminal work by Gatys et al., [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), many  -->