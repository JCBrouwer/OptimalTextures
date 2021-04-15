import torch
import torchvision
import torchvision.transforms.functional as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_styles(style_files, size, scale, oversize):
    styles = []
    for style_file in style_files:
        styles.append(load_image(style_file, size, scale, not oversize))
    return styles


def maybe_load_content(content_file, size):
    content = None
    if content_file is not None:
        content = load_image(content_file, size, no_quality_loss=False)
    return content


def load_image(path, size, scale=1, no_quality_loss=True):
    img = Image.open(path).convert(mode="RGB")

    size *= scale
    wpercent = size / float(img.size[0])
    hsize = int((float(img.size[1]) * float(wpercent)))

    if no_quality_loss:
        size = min(int(size), img.size[0])
        hsize = min(hsize, img.size[1])

    size = round32(size)
    hsize = round32(hsize)

    img = img.resize((int(size), hsize), Image.ANTIALIAS)

    return transforms.to_tensor(img).unsqueeze(0).to(device)


def round32(integer):
    return int(integer + 32 - 1) & -32


def save_image(output, args):
    outs = [name(style) for style in args.style]
    if len(args.style) > 1:
        outs += ["blend" + str(args.mixing_alpha)]
    if args.content is not None:
        outs += [name(args.content), "strength" + str(args.content_strength)]
    outs += [args.hist_mode + "hist"]
    if args.no_pca:
        outs += ["no_pca"]
    if args.no_multires:
        outs += ["no_multires"]
    if args.style_scale != 1:
        outs += ["scale" + str(args.style_scale)]
    if args.color_transfer is not None:
        outs += [args.color_transfer]
    outs += [str(args.size)]
    outname = "_".join(outs)
    torchvision.utils.save_image(output, f"{args.output_dir}/{outname}.png")


def name(filepath):
    return filepath.split("/")[-1].split(".")[0]
