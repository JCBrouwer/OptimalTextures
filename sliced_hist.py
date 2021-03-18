import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def swap_color_channel(target, source, colorspace="HSV"):  # YCbCr also works
    target_channels = list(target.convert(colorspace).split())
    source_channels = list(source.resize(target.size).convert(colorspace).split())
    target_channels[0] = source_channels[0]
    return Image.merge(colorspace, target_channels).convert("RGB")


def match_histogram(target, source, eps=1e-2):
    """Based on https://github.com/ProGamerGov/Neural-Tools/blob/master/linear-color-transfer.py#L36"""

    mu_t = target.mean((2, 3), keepdim=True)
    hist_t = (target - mu_t).reshape(target.size(1), -1)  # [b, c, h * w]
    cov_t = torch.mm(hist_t, hist_t.T) / hist_t.shape[1] + eps * torch.eye(hist_t.shape[0])
    eigval_t, eigvec_t = torch.symeig(cov_t, eigenvectors=True, upper=True)
    E_t = torch.sqrt(torch.diagflat(eigval_t))
    E_t[E_t != E_t] = 0  # Convert nan to 0
    Q_t = torch.mm(torch.mm(eigvec_t, E_t), eigvec_t.T)

    mu_s = source.mean((2, 3), keepdim=True)
    hist_s = (source - mu_s).reshape(source.size(1), -1)
    cov_s = torch.mm(hist_s, hist_s.T) / hist_s.shape[1] + eps * torch.eye(hist_s.shape[0])
    eigval_s, eigvec_s = torch.symeig(cov_s, eigenvectors=True, upper=True)
    E_s = torch.sqrt(torch.diagflat(eigval_s))
    E_s[E_s != E_s] = 0
    Q_s = torch.mm(torch.mm(eigvec_s, E_s), eigvec_s.T)

    matched = torch.mm(torch.mm(Q_s, torch.inverse(Q_t)), hist_t)
    matched = matched.reshape(*target.shape) + mu_s
    matched = matched.clamp(0, 1)

    return matched


def gram_schmidt(vv):
    """From https://github.com/legendongary/pytorch-gram-schmidt"""

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + vk.dot(uj) / uj.dot(uj) * uj
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu


def random_basis(N):
    vec = torch.randn(1, N)
    vec /= vec.norm()
    return gram_schmidt(vec)


def project(tensor, basis):
    return torch.mm(tensor.reshape(tensor.size(1), -1).t(), basis).reshape(tensor.shape)


def deproject(tensor, basis):
    return torch.div(tensor.reshape(tensor.size(1), -1).t(), basis).reshape(tensor.shape)


def optimal_transport(output, style, passes):
    N = output.shape[1]  # channels
    for _ in range(int(N / passes)):
        basis = random_basis(N)
        rotated_style = project(style, basis)
        rotated_output = project(output, basis)
        matched_output = match_histogram(rotated_output, rotated_style)
        output = deproject(matched_output, basis)
    return output


if __name__ == "__main__":
    contim = np.asarray(Image.open("../content/-166.jpg"))
    content = torch.from_numpy(contim).permute(2, 0, 1)[None, ...].float() / 255
    stylim = np.asarray(Image.open("../style/candy.jpg").resize((contim.shape[1], contim.shape[0])))
    style = torch.from_numpy(stylim).permute(2, 0, 1)[None, ...].float() / 255
    matched = (match_histogram(content, style) * 255)[0].permute(1, 2, 0).numpy().astype(np.uint8)
    Image.fromarray(np.concatenate((contim, stylim, matched), axis=1)).save("hist_match.png")

    contim = Image.open("../content/-166.jpg")
    stylim = Image.open("../style/candy.jpg")
    matched = swap_color_channel(contim, stylim).save("color_swap.png")
