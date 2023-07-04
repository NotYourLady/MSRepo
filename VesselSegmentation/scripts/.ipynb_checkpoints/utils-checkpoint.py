import numpy as np
import sys
import os
from time import time, sleep
import matplotlib.pyplot as plt

def print_img(vol, axis, slice=None, title= 'title', cmap='hot'):
    axis.set_title(title)
    im = axis.imshow(vol[:, :, slice], cmap=cmap)
    plt.colorbar(im)