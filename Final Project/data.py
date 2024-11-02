from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch
from PIL import Image
import numpy as np
import random
from itertools import permutations

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

class RandomChannelSwap(object):
    def __init__(self, probability):
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(f'Image should be PIL type. Got {type(image)}')
        if not _is_pil_image(depth):
            raise TypeError(f'Image should be PIL type. Got {type(depth)}')
        
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) -1)])])
        
        return {'image': image, 'depth': depth}
