import numpy as np
import torch
from PIL import Image
from torchinterp1d import Interp1d

interp1d = Interp1d()

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


def hist_match_np(target, source):
    matched = torch.empty_like(target.T, device=device)
    target = target.T.cpu().numpy()
    source = source.T.cpu().numpy()

    nc, npx = target.shape
    bins = 128

    for j in range(nc):
        lo = min(target[j].min(), source[j].min())
        hi = max(target[j].max(), source[j].max())

        target_hist, bin_edges = np.histogram(target[j], bins=bins, range=[lo, hi])
        source_hist, _ = np.histogram(source[j], bins=bins, range=[lo, hi])

        target_cdf = target_hist.cumsum().astype(np.float32)
        target_cdf /= target_cdf[-1]

        source_cdf = source_hist.cumsum().astype(np.float32)
        source_cdf /= source_cdf[-1]

        remapped_cdf = np.interp(target_cdf, source_cdf, bin_edges[1:])

        matched[j] = torch.from_numpy(np.interp(target[j], bin_edges[1:], remapped_cdf, left=0, right=bins)).to(device)

    return matched.T


def hist_match(target, source, bins=128):
    target = target.T
    source = source.T
    matched = torch.empty_like(target, device=device)
    for j in range(len(target)):
        lo = torch.min(target[j].min(), source[j].min())
        hi = torch.max(target[j].max(), source[j].max())

        target_hist = torch.histc(target[j], bins, lo, hi)
        source_hist = torch.histc(source[j], bins, lo, hi)

        target_cdf = target_hist.cumsum(0)
        target_cdf = target_cdf / target_cdf[-1]

        source_cdf = source_hist.cumsum(0)
        source_cdf = source_cdf / source_cdf[-1]

        bin_edges = torch.linspace(lo, hi, bins + 1, device=device)

        remapped_cdf = interp1d(source_cdf, bin_edges[1:], target_cdf).squeeze()
        # ^^^ first positions of this have -1000 values all of a sudden?!

        matched[j] = interp1d(bin_edges[1:], remapped_cdf, target[j])

    return matched.T


if __name__ == "__main__":
    import numpy as np
    import torch
    from torchinterp1d import Interp1d

    interp1d = Interp1d()

    # histogram matching with numpy

    random_state = np.random.RandomState(12345)

    bins = 64

    target = random_state.normal(size=(128 * 128)) * 2
    source = random_state.normal(size=(128 * 128)) * 2
    matched = np.empty_like(target)

    lo = min(target.min(), source.min())
    hi = max(target.max(), source.max())

    target_hist_np, bin_edges_np = np.histogram(target, bins=bins, range=[lo, hi])
    source_hist_np, _ = np.histogram(source, bins=bins, range=[lo, hi])

    target_cdf_np = target_hist_np.cumsum()
    target_cdf_np = target_cdf_np / target_cdf_np[-1]

    source_cdf_np = source_hist_np.cumsum()
    source_cdf_np = source_cdf_np / source_cdf_np[-1]

    remapped_cdf_np = np.interp(target_cdf_np, source_cdf_np, bin_edges_np[1:])

    matched_np = np.interp(target, bin_edges_np[1:], remapped_cdf_np, left=0, right=bins)

    # now with pytorch

    target = torch.from_numpy(target)
    source = torch.from_numpy(source)

    target_hist = torch.histc(target, bins, lo, hi)
    source_hist = torch.histc(source, bins, lo, hi)

    assert np.allclose(target_hist_np, target_hist.numpy())
    assert np.allclose(source_hist_np, source_hist.numpy())

    target_cdf = target_hist.cumsum(0)
    target_cdf = target_cdf / target_cdf[-1]

    assert np.allclose(target_cdf_np, target_cdf.numpy())

    source_cdf = source_hist.cumsum(0)
    source_cdf = source_cdf / source_cdf[-1]

    assert np.allclose(source_cdf_np, source_cdf.numpy())

    bin_edges = torch.linspace(lo, hi, bins + 1)

    assert np.allclose(bin_edges_np, bin_edges.numpy())

    remapped_cdf = interp1d(source_cdf, bin_edges[1:], target_cdf).squeeze()
    # ^^^ first positions of this have -100 values all of a sudden?!

    print(remapped_cdf_np)
    print(remapped_cdf.numpy())
    assert np.allclose(remapped_cdf_np, remapped_cdf.numpy())  # fails

    matched = interp1d(bin_edges[1:], remapped_cdf, target)

    assert np.allclose(matched_np, matched.numpy())
