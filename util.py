import torch
import torchvision
import torchvision.transforms.functional as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_styles(style_files, size):
    styles = []
    for style_file in style_files:
        styles.append(load_image(style_file, size))
    return styles


def maybe_load_content(content_file, size):
    content = None
    if content_file is not None:
        content = load_image(content_file, size)
    return content


def load_image(path, size=256):
    img = Image.open(path).convert(mode="RGB")

    wpercent = size / float(img.size[0])
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((int(size), hsize), Image.ANTIALIAS)

    return transforms.to_tensor(img).unsqueeze(0).to(device)


def save_image(output, args):
    outs = [name(style) for style in args.style]
    if len(args.style) > 1:
        outs += ["blend", str(args.mixing_alpha)]
    if args.content is not None:
        outs += [name(args.content), "strength", str(args.content_strength)]
    if args.hist_mode != "chol":
        outs += [args.hist_mode + "hist"]
    if args.no_pca:
        outs += ["no_pca"]
    if args.no_multires:
        outs += ["no_multires"]
    outs += [str(args.size)]
    outname = "_".join(outs)
    torchvision.utils.save_image(output, f"output/{outname}.png")


def name(filepath):
    return filepath.split("/")[-1].split(".")[0]
