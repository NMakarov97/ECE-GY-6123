from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch
from PIL import Image
import numpy as np

def _is_pil_image(img) -> bool:
    return isinstance(img, Image.Image)

def _is_numpy_image(img) -> bool:
    return isinstance(img, np.ndarray)
