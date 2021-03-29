import numpy as np
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def swap_color_channel(target, source, colorspace="HSV"):  # YCbCr also works
    target_channels = list(target.convert(colorspace).split())
    source_channels = list(source.resize(target.size).convert(colorspace).split())
    target_channels[0] = source_channels[0]
    return Image.merge(colorspace, target_channels).convert("RGB")


def hist_match(target, source, mode="chol", eps=1e-2):
    """From https://github.com/ProGamerGov/Neural-Tools/blob/master/linear-color-transfer.py#L36"""
    target = target.permute(0, 3, 1, 2)
    source = source.permute(0, 3, 1, 2)

    if mode == "cdf":
        b, c, h, w = target.shape
        matched = cdf_match(target.reshape(c, -1), source.reshape(c, -1)).reshape(b, c, h, w)

    else:
        mu_t = target.mean((2, 3), keepdim=True)
        hist_t = (target - mu_t).view(target.size(1), -1)  # [c, b * h * w]
        cov_t = hist_t @ hist_t.T / hist_t.shape[1] + eps * torch.eye(hist_t.shape[0], device=device)

        mu_s = source.mean((2, 3), keepdim=True)
        hist_s = (source - mu_s).view(source.size(1), -1)
        cov_s = hist_s @ hist_s.T / hist_s.shape[1] + eps * torch.eye(hist_s.shape[0], device=device)

        if mode == "chol":
            chol_t = torch.linalg.cholesky(cov_t)
            chol_s = torch.linalg.cholesky(cov_s)
            matched = chol_s @ torch.inverse(chol_t) @ hist_t

        elif mode == "pca":
            eva_t, eve_t = torch.symeig(cov_t, eigenvectors=True, upper=True)
            Qt = eve_t @ torch.sqrt(torch.diag(eva_t)) @ eve_t.T
            eva_s, eve_s = torch.symeig(cov_s, eigenvectors=True, upper=True)
            Qs = eve_s @ torch.sqrt(torch.diag(eva_s)) @ eve_s.T
            matched = Qs @ torch.inverse(Qt) @ hist_t

        elif mode == "sym":
            eva_t, eve_t = torch.symeig(cov_t, eigenvectors=True, upper=True)
            Qt = eve_t @ torch.sqrt(torch.diag(eva_t)) @ eve_t.T
            Qt_Cs_Qt = Qt @ cov_s @ Qt
            eva_QtCsQt, eve_QtCsQt = torch.symeig(Qt_Cs_Qt, eigenvectors=True, upper=True)
            QtCsQt = eve_QtCsQt @ torch.sqrt(torch.diag(eva_QtCsQt)) @ eve_QtCsQt.T
            matched = torch.inverse(Qt) @ QtCsQt @ torch.inverse(Qt) @ hist_t

        matched = matched.view(*target.shape) + mu_s

    return matched.permute(0, 2, 3, 1)


def interp(x, xp, fp):
    """
    from https://github.com/numpy/numpy/blob/main/numpy/core/src/multiarray/compiled_base.c#L489
    """
    f = torch.zeros_like(x)

    idxs = torch.searchsorted(xp, x)
    idxs1 = (idxs + 1).clamp(0, len(xp) - 1)

    slopes = (fp[idxs1] - fp[idxs]) / (xp[idxs1] - xp[idxs])
    f = slopes * (x - xp[idxs]) + fp[idxs]

    infinite = ~torch.isfinite(f)
    # f[infinite] = fp[idxs[infinite]]
    if infinite.any():
        infidxs1 = idxs1[infinite]
        f[infinite] = slopes[infinite] * (x[infinite] - xp[infidxs1]) + fp[infidxs1]

        still_infinite = ~torch.isfinite(f)
        if still_infinite.any():
            f[still_infinite] = fp[idxs[still_infinite]]

    return f


def cdf_match(target, source, bins=128):
    matched = torch.empty_like(target)
    for i, (target_channel, source_channel) in enumerate(zip(target, source)):
        lo = torch.min(target_channel.min(), source_channel.min())
        hi = torch.max(target_channel.max(), source_channel.max())

        target_hist = torch.histc(target_channel, bins, lo, hi)
        source_hist = torch.histc(source_channel, bins, lo, hi)
        bin_edges = torch.linspace(lo, hi, bins + 1, device=device)[1:]

        target_cdf = target_hist.cumsum(0)
        target_cdf = target_cdf / target_cdf[-1]

        source_cdf = source_hist.cumsum(0)
        source_cdf = source_cdf / source_cdf[-1]

        remapped_cdf = interp(target_cdf, source_cdf, bin_edges)
        matched[i] = interp(target_channel, bin_edges, remapped_cdf)
    return matched
