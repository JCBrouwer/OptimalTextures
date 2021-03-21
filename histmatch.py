import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pca_match(target, source, eps=1e-2):
    """Based on https://github.com/ProGamerGov/Neural-Tools/blob/master/linear-color-transfer.py#L36"""

    mu_t = target.mean((2, 3), keepdim=True)
    hist_t = (target - mu_t).reshape(target.size(1), -1)  # [c, b * h * w]
    cov_t = hist_t @ hist_t.T / hist_t.shape[1] + eps * torch.eye(hist_t.shape[0])

    eigval_t, eigvec_t = torch.symeig(cov_t, eigenvectors=True, upper=True)
    E_t = torch.sqrt(torch.diagflat(eigval_t))
    E_t[E_t != E_t] = 0  # Convert nan to 0
    Q_t = (eigvec_t @ E_t) @ eigvec_t.T

    mu_s = source.mean((2, 3), keepdim=True)
    hist_s = (source - mu_s).reshape(source.size(1), -1)
    cov_s = hist_s @ hist_s.T / hist_s.shape[1] + eps * torch.eye(hist_s.shape[0])

    eigval_s, eigvec_s = torch.symeig(cov_s, eigenvectors=True, upper=True)
    E_s = torch.sqrt(torch.diagflat(eigval_s))
    E_s[E_s != E_s] = 0
    Q_s = (eigvec_s @ E_s) @ eigvec_s.T

    matched = (Q_s @ torch.inverse(Q_t)) @ hist_t
    matched = matched.reshape(*target.shape) + mu_s
    matched = matched.clamp(0, 1)

    return matched


def cdf_match(target, source, shared_limits=True):
    """From https://sgugger.github.io/deep-painterly-harmonization.html"""

    b, c, h, w = target.shape
    n = h * w
    target = target.reshape(c, n)
    source = source.reshape(c, n)
    n_bins = 128

    if shared_limits:
        mins = torch.minimum(torch.min(target, 1)[0], torch.min(source, 1)[0])
        maxes = torch.minimum(torch.max(target, 1)[0], torch.max(source, 1)[0])
        source_hist = torch.stack([torch.histc(source[i], n_bins, mins[i], maxes[i]) for i in range(c)])
    else:
        source_hist = torch.stack([torch.histc(source[i], n_bins) for i in range(c)])

    _, sort_idx = target.data.sort(1)

    hist = source_hist * n / source_hist.sum(1).unsqueeze(1)
    cum_source = hist.cumsum(1)
    cum_prev = torch.cat([torch.zeros(c, 1).to(device), cum_source[:, :-1]], 1)

    rng = torch.arange(1, n + 1).unsqueeze(0).float().to(device)
    idx = (cum_source.unsqueeze(1) - rng.unsqueeze(2) < 0).sum(2).long()

    if shared_limits:
        ymin, ymax = mins.unsqueeze(1), maxes.unsqueeze(1)
    else:
        ymin, ymax = target.data.min(1)[0].unsqueeze(1), target.data.max(1)[0].unsqueeze(1)
    step = (ymax - ymin) / n_bins
    ratio = (rng - cum_prev.view(-1)[idx.view(-1)].view(c, -1)) / (1e-8 + hist.view(-1)[idx.view(-1)].view(c, -1))
    ratio = ratio.squeeze().clamp(0, 1)
    matched = ymin + (ratio + idx.float()) * step

    _, remap = sort_idx.sort()
    matched = matched.view(-1)[remap.view(-1)].view(c, -1)

    return matched.reshape(b, c, h, w)
