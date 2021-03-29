import torch
import torchvision
import torchvision.transforms.functional as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path, size=256):
    img = Image.open(path).convert(mode="RGB")

    wpercent = size / float(img.size[0])
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((int(size), hsize), Image.ANTIALIAS)

    return transforms.to_tensor(img).unsqueeze(0).to(device)


def save_image(output, args):
    outname = name(args.style)
    if args.content is not None:
        outname += name(args.content) + "_" + str(args.content_strength)
    outname += "_" + args.hist_mode
    if not args.no_pca:
        outname += "_pca"
    if not args.no_multires:
        outname += "_multires"
    torchvision.utils.save_image(output, f"output/{outname}.png")


def name(filepath):
    return filepath.split("/")[-1].split(".")[0]
