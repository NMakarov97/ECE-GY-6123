from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch
from PIL import Image

def _is_pil_image(img):
    return isinstance(img, Image.Image)
