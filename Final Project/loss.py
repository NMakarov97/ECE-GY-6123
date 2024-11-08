import torch
from torch import Tensor
import math
import torch.nn.functional as F

def gaussian(window_size:int, sigma:float) -> Tensor:
    gauss = Tensor(
        [
            math.exp(-((x - window_size//2)**2)/float(2*sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss/gauss.sum()

def create_window(window_size:int, channel:int=1) -> Tensor:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True) -> tuple[Tensor, Tensor]:
    L = val_range
    C1 = (0.01)**2
    C2 = (0.03)**2
    padd = 0
    (_, channel, height, width) = img1.size()

    # Create window if necessary
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel).to(img1.device)

    # Calculate mu for both images using gaussian filter
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Calculate sigma square
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return ret, cs
