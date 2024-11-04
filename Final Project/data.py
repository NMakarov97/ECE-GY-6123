from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch
from torch import Tensor
from PIL import Image
import numpy as np
import random
from io import BytesIO

def _is_pil_image(img) -> bool:
    return isinstance(img, Image.Image)

def _is_numpy_image(img) -> bool:
    return isinstance(img, np.ndarray)

class RandomHorizontalFlip(object):
    def __call__(self, sample:dict[str, Image.Image]) -> dict[str, Image.Image]:
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(f'Image should be PIL type. Got {type(image)}')
        if not _is_pil_image(depth):
            raise TypeError(f'Image should be PIL type. Got {type(depth)}')

        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}

class RandomChannelSwap(object):
    def __init__(self, probability:float) -> None:
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample:dict[str, Image.Image]) -> dict[str, Image.Image]:
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(f'Image should be PIL type. Got {type(image)}')
        if not _is_pil_image(depth):
            raise TypeError(f'Image should be PIL type. Got {type(depth)}')
        
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) -1)])])
        
        return {'image': image, 'depth': depth}

def loadZipToMem(zip_file:str) -> tuple[dict[str, bytes], list[list[str]]]:
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
    return data, nyu2_train # type: ignore

class depthDatasetMemory(Dataset):
    def __init__(self, data:dict[str, bytes], nyu2_train:list[list[str]], transform:transforms.Compose) -> None:
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform

    def __getitem__(self, idx:int) -> dict[str, Image.Image]:
        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1]]))
        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self) -> int:
        return len(self.nyu_dataset)

class ToTensor(object):
    def __init__(self, is_test:bool=False) -> None:
        self.is_test = is_test

    def __call__(self, sample:dict[str, Image.Image|np.ndarray]) -> dict[str, Tensor]:
        image, depth = sample['image'], sample['depth']

        image = self.to_tensor(image)

        depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000 # type: ignore
        else:
            depth = self.to_tensor(depth).float() * 1000 # type: ignore

        depth = torch.clamp(depth, 10, 1000)
        return {'image': image, 'depth': depth}

    def to_tensor(self, pic:Image.Image|np.ndarray) -> Tensor:
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                f'Picture should be PIL image or numpy array. Got {type(pic)}'
            )
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(255)

        # Case for PIL image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

def getNoTransform(is_test:bool=False) -> transforms.Compose:
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])

def getDefaultTrainTransform() -> transforms.Compose:
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])

def getTrainingTestingData(path: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    data, nyu2_train = loadZipToMem(path)
    transformed_training = depthDatasetMemory(
        data, nyu2_train, transform=getDefaultTrainTransform()
    )
    transformed_testing = depthDatasetMemory(
        data, nyu2_train, transform=getNoTransform()
    )
    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, batch_size, shuffle=False)
