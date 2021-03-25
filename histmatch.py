import numpy as np
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def swap_color_channel(target, source, colorspace="HSV"):  # YCbCr also works
    target_channels = list(target.convert(colorspace).split())
    source_channels = list(source.resize(target.size).convert(colorspace).split())
    target_channels[0] = source_channels[0]
    return Image.merge(colorspace, target_channels).convert("RGB")


def old_pca_match(target, source, eps=1e-2):
    """From https://github.com/ProGamerGov/Neural-Tools/blob/master/linear-color-transfer.py#L36"""

    mu_t = target.mean((2, 3), keepdim=True)
    hist_t = (target - mu_t).view(target.size(1), -1)  # [c, b * h * w]
    cov_t = hist_t @ hist_t.T / hist_t.shape[1] + eps * torch.eye(hist_t.shape[0], device=device)

    eigval_t, eigvec_t = torch.symeig(cov_t, eigenvectors=True, upper=True)
    E_t = torch.sqrt(torch.diagflat(eigval_t))
    E_t[E_t != E_t] = 0  # Convert nan to 0
    Q_t = (eigvec_t @ E_t) @ eigvec_t.T

    mu_s = source.mean((2, 3), keepdim=True)
    hist_s = (source - mu_s).view(source.size(1), -1)
    cov_s = hist_s @ hist_s.T / hist_s.shape[1] + eps * torch.eye(hist_s.shape[0], device=device)

    eigval_s, eigvec_s = torch.symeig(cov_s, eigenvectors=True, upper=True)
    E_s = torch.sqrt(torch.diagflat(eigval_s))
    E_s[E_s != E_s] = 0
    Q_s = (eigvec_s @ E_s) @ eigvec_s.T

    matched = (Q_s @ torch.inverse(Q_t)) @ hist_t
    matched = matched.view(*target.shape) + mu_s
    matched = matched.clamp(0, 1)

    return matched


def pca_match(target, source, eps=1e-2):
    """From https://github.com/ProGamerGov/Neural-Tools/blob/master/linear-color-transfer.py#L36"""
    npx, nc = target.shape
    # print(target.shape, source.shape)

    mu_t = target.mean(0)
    hist_t = (target - mu_t).T
    # print(hist_t.shape)
    cov_t = hist_t @ hist_t.T / npx + eps * torch.eye(nc, device=device)
    # print(cov_t.shape)

    eigval_t, eigvec_t = torch.symeig(cov_t, eigenvectors=True, upper=True)
    E_t = torch.sqrt(torch.diagflat(eigval_t))
    E_t[E_t != E_t] = 0  # Convert nan to 0
    Q_t = (eigvec_t @ E_t) @ eigvec_t.T
    # print(Q_t.shape)

    mu_s = source.mean(0)
    # print(mu_s.shape)
    hist_s = (source - mu_s).T
    cov_s = hist_s @ hist_s.T / npx + eps * torch.eye(nc, device=device)

    eigval_s, eigvec_s = torch.symeig(cov_s, eigenvectors=True, upper=True)
    E_s = torch.sqrt(torch.diagflat(eigval_s))
    E_s[E_s != E_s] = 0
    Q_s = (eigvec_s @ E_s) @ eigvec_s.T

    matched = (Q_s @ torch.inverse(Q_t)) @ hist_t
    # print(matched.shape)
    matched = matched.T.reshape(npx, nc) + mu_s
    # print(matched.shape)

    matched = matched.clamp(0, 1)

    return matched


def colour_transfer_mkl(x0, x1, eps=1e-2):
    """From https://github.com/ptallada/colour_transfer/blob/master/colour_transfer.py#L10"""
    a = torch.cov(x0.T)
    b = torch.cov(x1.T)

    Da2, Ua = torch.linalg.eig(a)
    Da = torch.diag(torch.sqrt(Da2.clip(eps, None)))

    C = torch.dot(torch.dot(torch.dot(torch.dot(Da, Ua.T), b), Ua), Da)

    Dc2, Uc = torch.linalg.eig(C)
    Dc = torch.diag(torch.sqrt(Dc2.clip(eps, None)))

    Da_inv = torch.diag(1.0 / (torch.diag(Da)))

    t = torch.dot(torch.dot(torch.dot(torch.dot(torch.dot(torch.dot(Ua, Da_inv), Uc), Dc), Uc.T), Da_inv), Ua.T)

    mx0 = torch.mean(x0, axis=0)
    mx1 = torch.mean(x1, axis=0)

    return torch.dot(x0 - mx0, t) + mx1


def cdf_match(target, source):
    """
    Expects two flattened tensors of shape (pixels, channels)

    From https://sgugger.github.io/deep-painterly-harmonization.html
    """
    target = target.T
    source = source.T

    nc, npx = target.shape
    nbin = 128
    # print(target.shape)

    mins = torch.minimum(torch.min(target, 1)[0], torch.min(source, 1)[0])
    maxes = torch.minimum(torch.max(target, 1)[0], torch.max(source, 1)[0])
    hist_ref = torch.stack([torch.histc(source[i], nbin, mins[i], maxes[i]) for i in range(nc)])

    _, sort_idx = target.data.sort(1)

    hist = hist_ref * npx / hist_ref.sum(1).unsqueeze(1)  # Normalization between the different lengths of masks.
    cum_ref = hist.cumsum(1)
    cum_prev = torch.cat([torch.zeros(nc, 1).cuda(), cum_ref[:, :-1]], 1)

    rng = torch.arange(1, npx + 1).unsqueeze(0).cuda()
    idx = (cum_ref.unsqueeze(1) - rng.unsqueeze(2) < 0).sum(2).long()

    step = (maxes - mins) / nbin
    ratio = (rng - cum_prev.view(-1)[idx.view(-1)].view(nc, -1)) / (1e-8 + hist.view(-1)[idx.view(-1)].view(nc, -1))
    ratio = ratio.squeeze().clamp(0, 1)
    matched = mins[:, None] + (ratio + idx.float()) * step[:, None]

    _, remap = sort_idx.sort()
    matched = matched.view(-1)[remap.view(-1)].view(nc, -1)

    return matched.T


def hist_match(target, source):
    matched = torch.empty_like(target.T, device=device)
    target = target.T.cpu().numpy()
    source = source.T.cpu().numpy()

    nc, npx = target.shape
    bins = 128

    for j in range(nc):
        lo = min(target[j].min(), source[j].min())
        hi = max(target[j].max(), source[j].max())

        p0r, edges = np.histogram(target[j], bins=bins, range=[lo, hi])
        p1r, _ = np.histogram(source[j], bins=bins, range=[lo, hi])

        cp0r = p0r.cumsum().astype(np.float32)
        cp0r /= cp0r[-1]

        cp1r = p1r.cumsum().astype(np.float32)
        cp1r /= cp1r[-1]

        f = np.interp(cp0r, cp1r, edges[1:])

        matched[j] = torch.from_numpy(np.interp(target[j], edges[1:], f, left=0, right=bins)).to(device)

    matched = matched.clamp(min(target.min(), source.min()), max(target.max(), source.max()))
    return matched.T
