import torch
import math
import torch.nn.functional as F

def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            math.exp(-((x - window_size//2)**2)/float(2*sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss/gauss.sum()
