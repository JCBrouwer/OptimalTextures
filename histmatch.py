import torch
from torch import Tensor


def hist_match(target: Tensor, source: Tensor, mode: str = "chol", eps: float = 1):
    target = target.permute(3, 0, 1, 2)
    source = source.permute(3, 0, 1, 2)
    c, b, h, w = target.shape

    if mode == "cdf":
        matched = cdf_match(target.view(c, -1), source.view(c, -1)).view(c, b, h, w)

    else:
        # based on https://github.com/ProGamerGov/Neural-Tools/blob/master/linear-color-transfer.py#L36

        mu_t = target.mean((2, 3), keepdim=True)
        hist_t = (target - mu_t).view(c, -1)  # [c, b * h * w]
        cov_t = hist_t @ hist_t.T / hist_t.shape[1] + eps * torch.eye(hist_t.shape[0], device=target.device)

        mu_s = source.mean((2, 3), keepdim=True)
        hist_s = (source - mu_s).view(c, -1)
        cov_s = hist_s @ hist_s.T / hist_s.shape[1] + eps * torch.eye(hist_s.shape[0], device=target.device)

        if mode == "chol":
            chol_t = torch.linalg.cholesky(cov_t)
            chol_s = torch.linalg.cholesky(cov_s)
            matched = chol_s @ torch.inverse(chol_t) @ hist_t

        elif mode == "pca":
            eva_t, eve_t = torch.linalg.eigh(cov_t, UPLO="U")
            Qt = eve_t @ torch.sqrt(torch.diag(eva_t)) @ eve_t.T
            eva_s, eve_s = torch.linalg.eigh(cov_s, UPLO="U")
            Qs = eve_s @ torch.sqrt(torch.diag(eva_s)) @ eve_s.T
            matched = Qs @ torch.inverse(Qt) @ hist_t

        else:  # mode == "sym"
            eva_t, eve_t = torch.linalg.eigh(cov_t, UPLO="U")
            Qt = eve_t @ torch.sqrt(torch.diag(eva_t)) @ eve_t.T
            Qt_Cs_Qt = Qt @ cov_s @ Qt
            eva_QtCsQt, eve_QtCsQt = torch.linalg.eigh(Qt_Cs_Qt, UPLO="U")
            QtCsQt = eve_QtCsQt @ torch.sqrt(torch.diag(eva_QtCsQt)) @ eve_QtCsQt.T
            matched = torch.inverse(Qt) @ QtCsQt @ torch.inverse(Qt) @ hist_t

        matched = matched.view(c, b, h, w) + mu_s

    return matched.permute(1, 2, 3, 0)


def cdf_match(target: Tensor, source: Tensor, bins: int = 256):
    matched = torch.empty_like(target)
    for i, (target_channel, source_channel) in enumerate(zip(target.contiguous(), source)):
        lo = torch.min(target_channel.min(), source_channel.min())
        hi = torch.max(target_channel.max(), source_channel.max())

        # TODO find batched method of getting histogram? maybe based on numpy's impl?
        # https://github.com/numpy/numpy/blob/v1.20.0/numpy/lib/histograms.py#L678
        target_hist = torch.histc(target_channel, bins, lo, hi)
        source_hist = torch.histc(source_channel, bins, lo, hi)
        bin_edges = torch.linspace(lo, hi, bins + 1, device=target.device)[1:]

        target_cdf = target_hist.cumsum(0)
        target_cdf = target_cdf / target_cdf[-1]

        source_cdf = source_hist.cumsum(0)
        source_cdf = source_cdf / source_cdf[-1]

        remapped_cdf = interp(target_cdf, source_cdf, bin_edges)
        matched[i] = interp(target_channel, bin_edges, remapped_cdf)
    return matched


def interp(x: Tensor, xp: Tensor, fp: Tensor):
    # based on https://github.com/numpy/numpy/blob/main/numpy/core/src/multiarray/compiled_base.c#L489

    f = torch.zeros_like(x)

    idxs = torch.searchsorted(xp, x)
    idxs_next = (idxs + 1).clamp(0, len(xp) - 1)

    slopes = (fp[idxs_next] - fp[idxs]) / (xp[idxs_next] - xp[idxs])
    f = slopes * (x - xp[idxs]) + fp[idxs]

    infinite = ~torch.isfinite(f)
    if infinite.any():
        inf_idxs_next = idxs_next[infinite]
        f[infinite] = slopes[infinite] * (x[infinite] - xp[inf_idxs_next]) + fp[inf_idxs_next]

        still_infinite = ~torch.isfinite(f)
        if still_infinite.any():
            f[still_infinite] = fp[idxs[still_infinite]]

    return f
