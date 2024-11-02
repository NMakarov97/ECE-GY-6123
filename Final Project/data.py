from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch
from PIL import Image
import numpy as np
import random

def _is_pil_image(img) -> bool:
    return isinstance(img, Image.Image)

def _is_numpy_image(img) -> bool:
    return isinstance(img, np.ndarray)

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(f'Image should be PIL type. Got {type(image)}')
        if not _is_pil_image(depth):
            raise TypeError(f'Image should be PIL type. Got {type(depth)}')

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}
