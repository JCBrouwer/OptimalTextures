import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as transforms

device = torch.device("cpu")  # "cuda" if torch.cuda.is_available() else "cpu")


def load_image(path):
    img = Image.open(path).convert(mode="RGB")
    return transforms.to_tensor(img).unsqueeze(0).to(device)


def save_image(path, tensor):
    pass
