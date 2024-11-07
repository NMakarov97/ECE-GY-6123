import matplotlib
import matplotlib.cm as cm
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch import Tensor
from os import listdir, path

def DepthNorm(depth:Tensor, maxDepth:float=1000.0) -> Tensor:
    return maxDepth / depth

class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val:float, n:int=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def colorize(value, vmin:float=10, vmax:float=1000, cmap:str='plasma') -> np.ndarray:
    value = value.cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        # Avoid 0-division
        value = value*0.0

    cmapper = cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:,:,:3]

    return img.transpose((2,0,1))

# def load_from_checkpoint(ckpt, model, optimizer, epochs, loss_meter=None):
#     checkpoint = torch.load(ckpt)
#     ckpt_epoch = epochs - (checkpoint['epoch'] + 1)
#     if ckpt_epoch <= 0:
#         raise ValueError(
#             f'Epochs provided: {epochs}, epochs completed in checkpoint: {checkpoint['epoch'] + 1}'
#         )
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optim_state_dict'])
#     return model, optimizer, ckpt_epoch

def init_or_load_model(depthmodel, enc_pretrain:bool, epochs:int, lr:float, ckpt:str|None=None, device:torch.device=torch.device('cuda:0'), loss_meter=None) -> tuple[nn.Module, torch.optim.Adam, int]:
    
    if ckpt is not None:
        files = sorted(listdir(ckpt))
        checkpoint = torch.load(path.join(ckpt,files[-1]))
    
    model = depthmodel(encoder_pretrained=enc_pretrain)

    if ckpt is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if ckpt is not None:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
    
    start_epoch = 0
    if ckpt is not None:
        start_epoch = checkpoint['epoch'] + 1
        if start_epoch <= 0:
            raise ValueError(f'Epochs provided: {epochs}')

    return model, optimizer, start_epoch

def load_images(image_files:list[str]) -> np.ndarray:
    loaded_images = []
    for file in image_files:
        x = np.clip(np.asarray(Image.open(file).resize((640,480)), dtype=float) / 255, 0, 1).transpose(2, 0, 1)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)
