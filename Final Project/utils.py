import matplotlib
import matplotlib.cm as cm
import numpy as np

def DepthNorm(depth:float, maxDepth:float=1000.0) -> float:
    return maxDepth / depth

class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
