from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch
from PIL import Image
import numpy as np
import random
from io import BytesIO

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
        from itertools import permutations
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

def loadZipToMem(zip_file:str):
    from zipfile import ZipFile
    from sklearn.utils import shuffle
    # Load zip file to memory
    print('Loading dataset from zip file...', end='')

    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list(
        (
            row.split(',')
            for row in (data['data/nyu2_train.csv']).decode('utf-8').split('\n')
            if len(row) > 0
        )
    )

    nyu2_train = shuffle(nyu2_train, random_state=0)

    if nyu2_train:
        print(f'Loaded {len(nyu2_train)}')
    return data, nyu2_train

class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1]]))
        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return len(self.nyu_dataset)
