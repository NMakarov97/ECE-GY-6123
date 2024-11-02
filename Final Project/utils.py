import matplotlib
import matplotlib.cm as cm
import numpy as np

def DepthNorm(depth:float, maxDepth:float=1000.0) -> float:
    return maxDepth / depth
