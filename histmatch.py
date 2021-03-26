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


def hist_match(target, source, mode="chol", eps=1e-2):
    """From https://github.com/ProGamerGov/Neural-Tools/blob/master/linear-color-transfer.py#L36"""

    # if mode == "cdf":
    #     return cdf_match_np(target, source)
    # if mode == "mkl":
    #     return mkl_match(target, source)

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

    return matched


#### vvv        hist match graveyard        vvv


def cdf_match_np(target, source, bins=128):
    target = target.squeeze().permute(1, 2, 0)
    shape = target.shape
    target = target.reshape(-1, target.shape[1]).T.cpu().numpy()

    source = source.squeeze().permute(1, 2, 0).reshape(-1, target.shape[0]).T.cpu().numpy()

    matched = np.empty_like(target)

    for j in range(target.shape[0]):
        lo = min(target[j].min(), source[j].min())
        hi = max(target[j].max(), source[j].max())

        target_hist, bin_edges = np.histogram(target[j], bins=bins, range=[lo, hi])
        source_hist, _ = np.histogram(source[j], bins=bins, range=[lo, hi])

        target_cdf = target_hist.cumsum().astype(np.float32)
        target_cdf /= target_cdf[-1]

        source_cdf = source_hist.cumsum().astype(np.float32)
        source_cdf /= source_cdf[-1]

        remapped_cdf = np.interp(target_cdf, source_cdf, bin_edges[1:])

        matched[j] = np.interp(target[j], bin_edges[1:], remapped_cdf, left=0, right=bins)

    return torch.from_numpy(matched.T.reshape(1, *reversed(shape))).to(device)


def mkl_match(target, source, eps=1e-2):
    """From https://github.com/ptallada/colour_transfer/blob/master/colour_transfer.py#L10"""

    cov_t = target.T @ target / target.shape[1]
    cov_s = source.T @ source / source.shape[1]

    eva_t, eve_t = torch.linalg.eig(cov_t)
    eva_t = torch.diag(torch.sqrt(eva_t.clip(eps, None)))

    C = eva_t @ eve_t.T @ cov_s @ eve_t @ eva_t

    eva_c, eve_c = torch.linalg.eig(C)
    eva_c = torch.diag(torch.sqrt(eva_c.clip(eps, None)))

    eva_t_inv = torch.diag(1.0 / (torch.diag(eva_t)))

    t = eve_t @ eva_t_inv @ eve_c @ eva_c @ eve_c.T @ eva_t_inv @ eve_t.T

    mu_t = torch.mean(target, axis=0)
    mu_s = torch.mean(source, axis=0)

    return (target - mu_t) @ t + mu_s


def cdf_match_vec(target, source):
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


def cdf_match_pth(target, source, bins=128):
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

        remapped_cdf = interp1d(source_cdf, bin_edges[:-1], target_cdf).squeeze()
        # ^^^ first positions of this have values of -1000 all of a sudden?!

        # derp fix
        orders_of_magnitude = torch.sign(remapped_cdf) * torch.log10(torch.abs(remapped_cdf))
        broken = orders_of_magnitude < -2
        remapped_cdf[broken] = remapped_cdf[torch.min(broken, 0)[1]].clone()

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
