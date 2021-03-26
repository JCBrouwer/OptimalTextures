import torch
from PIL import Image
import torchvision.transforms.functional as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path, size=256):
    img = Image.open(path).convert(mode="RGB")

    wpercent = size / float(img.size[0])
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((int(size), hsize), Image.ANTIALIAS)

    return transforms.to_tensor(img).unsqueeze(0).to(device)


def name(filepath):
    return filepath.split("/")[-1].split(".")[0]