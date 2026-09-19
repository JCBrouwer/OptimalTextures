import argparse
from time import time
from typing import List, Optional

import torch
from kornia.color.hls import hls_to_rgb, rgb_to_hls
from torch import Tensor
from torch.nn.functional import interpolate

from histmatch import LINEAR_MODES, hist_match, quantiles
from util import get_iters_and_sizes, get_size, load_styles, maybe_load_content, resize, save_image, to_nchw, to_nhwc
from vgg import Decoder, Encoder


class OptimalTexture(torch.nn.Module):
    def __init__(
        self,
        size: int = 512,
        iters: int = 500,
        passes: int = 5,
        hist_mode: str = "sort",
        color_transfer: Optional[str] = None,
        content_strength: float = 0.1,
        style_scale: float = 1,
        mixing_alpha: float = 0.5,
        no_pca: bool = False,
        no_multires: bool = False,
        pca_variance: float = 0.99,
    ):
        super().__init__()

        self.hist_mode = hist_mode
        self.color_transfer = color_transfer
        self.content_strength = content_strength
        self.style_scale = style_scale
        self.mixing_alpha = mixing_alpha
        self.use_pca = not no_pca
        self.pca_variance = pca_variance

        # get number of iterations and sizes for optization
        self.passes = passes
        self.iters_per_pass_and_layer, self.sizes = get_iters_and_sizes(size, iters, passes, not no_multires)

        self.depths = list(range(5, 0, -1))  # relu5_1 -> relu1_1
        self.encoder = Encoder()
        self.decoders = torch.nn.ModuleList([Decoder(depth) for depth in self.depths])

    def encode_inputs(self, pastiche: Tensor, styles: List[Tensor], content: Optional[Tensor], size: int):
        # ensure pastiche, styles, and content are the correct size
        style_tens = [resize(s, size=get_size(size, self.style_scale, s.shape[2], s.shape[3])) for s in styles]
        if content is not None:
            cont_size = get_size(size, 1.0, content.shape[2], content.shape[3], oversize=True)
            cont_pyramid = self.encoder.pyramid(resize(content, size=cont_size))
        else:
            cont_size = (size, size)
            cont_pyramid = None
        pastiche = resize(pastiche, size=cont_size)

        # encode inputs to VGG feature space, all layers in a single pass
        style_pyramids = [self.encoder.pyramid(style) for style in style_tens]

        style_features, bases, content_features = [], [], []
        for depth in self.depths:
            style_feature = torch.cat([pyramid[depth - 1] for pyramid in style_pyramids])
            basis = PCA(style_feature, self.pca_variance) if self.use_pca else Identity(style_feature)
            style_features.append(basis.project(style_feature))
            bases.append(basis)

            if cont_pyramid is not None:
                # center content features on the style features, then express them in the same basis
                content_feature = cont_pyramid[depth - 1]
                content_feature = content_feature - content_feature.mean((0, 1, 2)) + basis.mean
                content_features.append(basis.project(content_feature))

        return pastiche, style_features, bases, content_features

    def forward(
        self,
        pastiche: Tensor,
        styles: List[Tensor],
        content: Optional[Tensor] = None,
        verbose: bool = False,
    ):
        for p in range(self.passes):
            if verbose:
                print(f"Pass {p}, size {self.sizes[p]}")

            # get style and content target features
            pastiche, style_features, bases, content_features = self.encode_inputs(
                pastiche, styles, content, self.sizes[p]
            )

            if len(styles) > 1:
                mixing_mask = torch.ceil(
                    torch.rand(style_features[1].shape[1:3], device=pastiche.device) - self.mixing_alpha
                )[None, None, ...]
                style_features = mix_style_features(style_features, mixing_mask, self.mixing_alpha, self.hist_mode)

            for l, (depth, decoder) in enumerate(zip(self.depths, self.decoders)):
                if verbose:
                    print(f"Layer: relu{depth}_1")

                # encode to VGG feature space and project onto the style's principal components
                pastiche_feature = bases[l].project(self.encoder(pastiche, depth))

                iters = self.iters_per_pass_and_layer[p][l]
                if self.hist_mode in LINEAR_MODES and len(content_features) == 0:
                    iters = 1  # a linear match is exact after one step, see optimal_transport()

                for _ in range(iters):
                    pastiche_feature = optimal_transport(pastiche_feature, style_features[l], self.hist_mode)

                    if len(content_features) > 0 and l <= 2:  # apply content matching step
                        strength = self.content_strength / 2 ** (4 - l)  # 1, 2, or 4 depending on feature depth
                        pastiche_feature += strength * (content_features[l] - pastiche_feature)

                pastiche = decoder(bases[l].unproject(pastiche_feature))  # decode back to image space

        if self.color_transfer is not None:
            assert content is not None, "Color transfer requires content image"
            target_hls = rgb_to_hls(content)
            target_hls[:, 1] = rgb_to_hls(pastiche)[:, 1]  # swap lightness channel
            target = hls_to_rgb(target_hls)

            if self.color_transfer == "opt":
                pastiche, target = to_nhwc(pastiche), to_nhwc(target)
                for _ in range(3):
                    pastiche = optimal_transport(pastiche, target, "sort")
                pastiche = to_nchw(pastiche)

            elif self.color_transfer == "lum":
                pastiche = target  # return pastiche with hue and saturation from content

        return pastiche


def random_rotation(N: int, device="cpu", dtype=torch.float32):
    """
    Draws a uniformly distributed random N-dimensional orthogonal matrix (inverse = transpose): the Q of the QR
    decomposition of a Gaussian matrix, with the signs fixed so that R has a positive diagonal (Mezzadri, 2007).
    Half of these are reflections rather than rotations, which makes no difference to histogram matching.
    """
    Q, R = torch.linalg.qr(torch.randn(N, N, device=device, dtype=dtype))
    return Q * torch.sign(R.diagonal())


def optimal_transport(pastiche_feature: Tensor, style_feature: Tensor, hist_mode: str):
    if hist_mode in LINEAR_MODES:
        # These match the full covariance, which no rotation changes: rotating first gives the same result, and
        # a second step finds the covariances already equal.
        return hist_match(pastiche_feature, style_feature, mode=hist_mode)

    rotation = random_rotation(pastiche_feature.shape[-1], pastiche_feature.device, pastiche_feature.dtype)

    rotated_pastiche = pastiche_feature @ rotation
    rotated_style = style_feature @ rotation

    matched_pastiche = hist_match(rotated_pastiche, rotated_style, mode=hist_mode)

    pastiche_feature = matched_pastiche @ rotation.T  # rotate back to normal

    return pastiche_feature


class PCA:
    """Principal components of [..., c] features that together explain the given fraction of their variance"""

    def __init__(self, tensor: Tensor, variance: float = 0.9):
        flat = tensor.reshape(-1, tensor.shape[-1])
        self.mean = flat.mean(0)
        centered = flat - self.mean

        # eigenvectors of the [c, c] covariance rather than an SVD of all [n, c] features: same components, but time
        # and memory no longer grow with the image size
        eigvals, eigvecs = torch.linalg.eigh(centered.T @ centered / flat.shape[0])
        eigvals, eigvecs = eigvals.flip(0).clamp_min(0), eigvecs.flip(1)  # largest first

        k = int((torch.cumsum(eigvals / eigvals.sum(), dim=0) < variance).sum()) + 1
        self.eigvecs = eigvecs[:, :k]

    def project(self, tensor: Tensor):
        return (tensor - self.mean) @ self.eigvecs

    def unproject(self, tensor: Tensor):
        return tensor @ self.eigvecs.T + self.mean


class Identity:
    """Stand-in for PCA that leaves features as they are"""

    def __init__(self, tensor: Tensor):
        self.mean = tensor.mean(tuple(range(tensor.dim() - 1)))

    def project(self, tensor: Tensor):
        return tensor

    def unproject(self, tensor: Tensor):
        return tensor


def sliced_wasserstein_loss(features: Tensor, sorted_style: Tensor, rotation: Tensor):
    """
    Squared distance between the sorted projections of features and style, i.e. how far hist_match would have to move
    the features along every axis of the rotation. From "A Sliced Wasserstein Loss for Neural Texture Synthesis"
    (Heitz et al., 2021). sorted_style is already rotated, sorted and resampled to the number of feature samples.
    """
    b, h, w, c = features.shape
    projected = (features.reshape(b, h * w, c) @ rotation).transpose(1, 2)
    return (projected.sort(dim=-1).values - sorted_style).square().mean()


def refine(
    pastiche: Tensor,
    styles: List[Tensor],
    encoder: Encoder,
    steps: int = 100,
    lr: float = 0.02,
    content_strength: float = 0.0,
    style_scale: float = 1.0,
    verbose: bool = False,
):
    """
    Polish the decoded image by gradient descent on the same objective, through the encoder only. The decoders are
    what limits the sharpness of the feed-forward result: they were trained to invert VGG features on photographs and
    blur whatever they have not seen. Starting from the optimal transport result, a few dozen steps are enough.
    """
    assert len(styles) == 1, "Refinement needs a single style image to compare with"
    encoder.requires_grad_(False)

    with torch.no_grad():
        size = pastiche.shape[-2:]
        style_pyramid = encoder.pyramid(resize(styles[0], size=get_size(size[0], style_scale, *styles[0].shape[2:])))
        anchor = encoder(pastiche, 4) if content_strength > 0 else None  # keeps the content's structure in place

    pastiche = pastiche.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pastiche], lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pyramid = encoder.pyramid(pastiche)
        loss = pastiche.new_zeros(())
        for features, style_features in zip(pyramid, style_pyramid):
            c = features.shape[-1]
            rotation = random_rotation(c, features.device, features.dtype)
            with torch.no_grad():
                sorted_style = (style_features.reshape(-1, c) @ rotation).T.sort(dim=-1).values
                sorted_style = quantiles(sorted_style, features.shape[1] * features.shape[2])
            loss = loss + sliced_wasserstein_loss(features, sorted_style, rotation)
        if anchor is not None:
            loss = loss + content_strength * (pyramid[3] - anchor).square().mean()
        loss.backward()
        optimizer.step()
        schedule.step()
        with torch.no_grad():
            pastiche.clamp_(0, 1)
        if verbose and step % 10 == 0:
            print(f"Refine step {step}, loss {loss.item():.4f}")

    return pastiche.detach()


def mix_style_features(style_features: List[Tensor], mixing_mask: Tensor, mixing_alpha: float, hist_mode: str):
    i = mixing_alpha

    for l, sf in enumerate(style_features):
        mix = to_nhwc(interpolate(mixing_mask, size=sf.shape[1:3], mode="nearest"))

        A, B = sf[[0]], sf[[1]]
        AtoB = hist_match(A, B, mode=hist_mode)
        BtoA = hist_match(B, A, mode=hist_mode)

        style_target = (A * (1 - i) + AtoB * i) * mix + (BtoA * (1 - i) + B * i) * (1 - mix)

        style_features[l] = style_target
    return style_features


if __name__ == "__main__":

    def required_length(nmin, nmax):
        class RequiredLength(argparse.Action):
            def __call__(self, parser, args, values, option_string=None):
                if not nmin <= len(values) <= nmax:
                    msg = f'argument "{self.dest}" requires between {nmin} and {nmax} arguments'
                    raise argparse.ArgumentTypeError(msg)
                setattr(args, self.dest, values)

        return RequiredLength

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--style", type=str, nargs="+", action=required_length(1, 2), default=["style/graffiti.jpg"], help="Example(s) of the style your texture should take")
    parser.add_argument("-c", "--content", type=str, default=None, help="The structure/shape you want your image to take")
    parser.add_argument("--batch", type=int, default=1, help="Batch size of images to generate")
    parser.add_argument("--size", type=int, default=512, help="The output size of the image (larger output = more memory/time required)")
    parser.add_argument("--passes", type=int, default=5, help="Number of times to loop over each of the 5 layers in VGG-19")
    parser.add_argument("--iters", type=int, default=500, help="Total number of iterations to optimize.")
    parser.add_argument("--hist_mode", type=str, choices=["sort", "cdf", "chol", "pca", "sym"], default="sort", help="Histogram matching strategy. sort is exact optimal transport per channel, cdf approximates it with binned histograms (faster for large images on CPU). chol, pca and sym only match mean and covariance: much faster, lower quality.")
    parser.add_argument("--color_transfer", type=str, default=None, choices=["lum", "opt"], help="Strategy to employ to keep original color of content image.")
    parser.add_argument("--content_strength", type=float, default=0.01, help="Strength with which to focus on the structure in your content image.")
    parser.add_argument("--style_scale", type=float, default=1.0, help="Scale the style relative to the generated image. Will affect the scale of details generated.")
    parser.add_argument("--mixing_alpha", type=float, default=0.5, help="Value between 0 and 1 for interpolation between 2 textures")
    parser.add_argument("--no_pca", action="store_true", help="Disable PCA of features (slower).")
    parser.add_argument("--pca_variance", type=float, default=0.99, help="Fraction of the style features' variance that the principal components should explain.")
    parser.add_argument("--no_multires", action="store_true", help="Disable multi-scale rendering (slower, less long-range texture qualities).")
    parser.add_argument("--refine", type=int, default=0, help="Number of gradient descent steps on a sliced Wasserstein loss to sharpen the result with afterwards (slower, higher quality). 100 is a good start.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random number generator.")
    parser.add_argument("--no_tf32", action="store_true", help="Disable tf32 format (probably slower).")
    parser.add_argument("--cudnn_benchmark", action="store_true", help="Enable CUDNN benchmarking (probably slower unless doing a high number of iterations).")
    parser.add_argument("--compile", action="store_true", help="Use PyTorch 2.0 compile function to optimize the model.")
    parser.add_argument("--device", type=str, default=None, help="Which device to run on.")
    parser.add_argument("--memory_format", type=str, default="contiguous", choices=["contiguous", "channels_last"], help="Which memory format to use for optimization.")
    parser.add_argument("--output_dir", type=str, default="output/", help="Directory to output results.")
    args = parser.parse_args()
    # fmt: on

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = not args.no_tf32
    torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
    memory_format = torch.contiguous_format if args.memory_format == "contiguous" else torch.channels_last
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    with torch.no_grad():
        styles = load_styles(
            args.style, size=args.size, scale=args.style_scale, device=device, memory_format=memory_format
        )
        if len(styles) > 1:
            assert styles[0].shape == styles[1].shape, "Style images must have the same shape"
        content = maybe_load_content(args.content, size=args.size, device=device, memory_format=memory_format)
        pastiche = torch.rand(content.shape if content is not None else (args.batch, 3, args.size, args.size)).to(
            device=device, memory_format=memory_format
        )

        texturizer = OptimalTexture(
            size=args.size,
            iters=args.iters,
            passes=args.passes,
            hist_mode=args.hist_mode,
            color_transfer=args.color_transfer,
            content_strength=args.content_strength,
            style_scale=args.style_scale,
            mixing_alpha=args.mixing_alpha,
            no_pca=args.no_pca,
            no_multires=args.no_multires,
            pca_variance=args.pca_variance,
        ).to(pastiche)

        if args.compile:
            texturizer = torch.compile(texturizer)

        t = time()
        pastiche = texturizer.forward(pastiche, styles, content, verbose=True)

    if args.refine > 0:
        strength = args.content_strength if content is not None else 0.0
        pastiche = refine(
            pastiche, styles, texturizer.encoder, args.refine, 0.02, strength, args.style_scale, verbose=True
        )
    print("Took:", time() - t)

    save_image(pastiche, args)
