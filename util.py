import torch
from PIL import Image
import torchvision.transforms.functional as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path):
    img = Image.open(path).convert(mode="RGB").resize((256, 256))
    return transforms.to_tensor(img).unsqueeze(0).to(device)