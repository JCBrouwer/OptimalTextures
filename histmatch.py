import torch
from torch import Tensor

LINEAR_MODES = ("chol", "pca", "sym")


def hist_match(target: Tensor, source: Tensor, mode: str = "sort", eps: float = 1e-2, bins: int = 256):
    """
    Match the per-channel distribution of target to that of source.

    target: [b, h, w, c], each sample in the batch is matched on its own
    source: [b', h', w', c], pooled into a single distribution

    "sort" and "cdf" match each channel's full histogram (the 1D optimal transport map). "sort" is exact, "cdf"
    approximates it with a binned histogram in linear time. "chol", "pca" and "sym" match mean and covariance of all
    channels at once with a linear map.
    """
    b, h, w, c = target.shape
    t = target.reshape(b, h * w, c).transpose(1, 2)  # [b, c, n]
    s = source.reshape(-1, c).T  # [c, m]

    if mode == "sort":
        matched = sort_match(t, s)
    elif mode == "cdf":
        matched = cdf_match(t, s, bins)
    else:
        matched = linear_match(t, s, mode, eps)

    return matched.transpose(1, 2).reshape(b, h, w, c)


def quantiles(sorted_values: Tensor, n: int):
    """Resample sorted values [..., m] to n evenly spaced quantiles [..., n] with linear interpolation"""
    m = sorted_values.shape[-1]
    if m == n:
        return sorted_values
    position = torch.linspace(0, m - 1, n, device=sorted_values.device)
    lower = position.floor().long().clamp(max=m - 2)
    frac = (position - lower).to(sorted_values.dtype)
    return torch.lerp(sorted_values[..., lower], sorted_values[..., lower + 1], frac)


def sort_match(target: Tensor, source: Tensor):
    """
    Exact 1D optimal transport for every channel at once: the k-th smallest target value becomes the k-th smallest
    source value (or the corresponding quantile when the two have a different number of samples).

    target: [b, c, n], source: [c, m]
    """
    source_quantiles = quantiles(source.sort(dim=-1).values, target.shape[-1])
    ranks = target.argsort(dim=-1)
    return torch.empty_like(target).scatter_(-1, ranks, source_quantiles.expand_as(target).contiguous())


def cdf_match(target: Tensor, source: Tensor, bins: int = 256):
    """
    Binned histogram matching for every channel at once. Each channel gets its own bin range. All histograms are
    counted by a single scatter_add, and because the bins are evenly spaced the look-up of a value's bin is a
    multiplication rather than a search.

    target: [b, c, n], source: [c, m]
    """
    b, c, n = target.shape
    source = source.expand(b, -1, -1)

    lo = torch.minimum(target.amin(-1, keepdim=True), source.amin(-1, keepdim=True))
    hi = torch.maximum(target.amax(-1, keepdim=True), source.amax(-1, keepdim=True))
    width = ((hi - lo) / bins).clamp_min(torch.finfo(target.dtype).tiny)

    def cdf(values):
        position = (values - lo) / width
        index = position.long().clamp_(0, bins - 1)
        counts = torch.zeros(b, c, bins, dtype=values.dtype, device=values.device)
        counts.scatter_add_(-1, index, torch.ones_like(values))
        cumulative = counts.cumsum(-1) / values.shape[-1]
        zero = torch.zeros_like(cumulative[..., :1])
        return torch.cat((zero, cumulative), -1), position  # CDF at the bins + 1 edges

    target_cdf, position = cdf(target)
    source_cdf, _ = cdf(source)

    # value at which the source CDF reaches the target CDF of each bin edge: inverse of the piecewise-linear source CDF
    upper = torch.searchsorted(source_cdf, target_cdf.contiguous()).clamp_(1, bins)
    cdf_lower, cdf_upper = source_cdf.gather(-1, upper - 1), source_cdf.gather(-1, upper)
    frac = ((target_cdf - cdf_lower) / (cdf_upper - cdf_lower).clamp_min(1e-12)).clamp_(0, 1)
    remapped_edges = lo + (upper - 1 + frac) * width

    # interpolate every target value between the remapped edges of its bin
    index = position.long().clamp_(0, bins - 1)
    frac = (position - index).clamp_(0, 1)
    return torch.lerp(remapped_edges.gather(-1, index), remapped_edges.gather(-1, index + 1), frac)


def linear_match(target: Tensor, source: Tensor, mode: str = "chol", eps: float = 1e-2):
    """
    Match mean and covariance with a linear map.
    Based on https://github.com/ProGamerGov/Neural-Tools/blob/master/linear-color-transfer.py#L36

    target: [b, c, n], source: [c, m]
    """
    eye = eps * torch.eye(target.shape[1], device=target.device, dtype=target.dtype)

    mu_t = target.mean(-1, keepdim=True)
    hist_t = target - mu_t
    cov_t = hist_t @ hist_t.transpose(1, 2) / hist_t.shape[-1] + eye

    mu_s = source.mean(-1, keepdim=True)
    hist_s = source - mu_s
    cov_s = hist_s @ hist_s.T / hist_s.shape[-1] + eye

    if mode == "chol":
        chol_t = torch.linalg.cholesky(cov_t)
        chol_s = torch.linalg.cholesky(cov_s)
        matched = chol_s @ torch.linalg.solve_triangular(chol_t, hist_t, upper=False)

    elif mode == "pca":
        matched = sqrtm(cov_s) @ torch.linalg.solve(sqrtm(cov_t), hist_t)

    elif mode == "sym":  # the optimal transport map between two Gaussians
        Qt = sqrtm(cov_t)
        Qt_inv = torch.linalg.inv(Qt)
        matched = Qt_inv @ sqrtm(Qt @ cov_s @ Qt) @ Qt_inv @ hist_t

    else:
        raise ValueError(f"Unknown histogram matching mode: {mode}")

    return matched + mu_s


def sqrtm(matrix: Tensor):
    """Square root of a symmetric positive semi-definite matrix"""
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    return (eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
