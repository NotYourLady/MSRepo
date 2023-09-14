import numpy as np
import sys
import os
from time import time, sleep
import matplotlib.pyplot as plt

def print_img(vol, axis, slice_=None, title= 'title', cmap='hot', bar=True):
    axis.set_title(title)
    im = axis.imshow(vol[:, :, slice_], cmap=cmap)
    if bar:
        plt.colorbar(im)

    
def print_imgs(list_imgs, slice_n=None, size=(5, 4)):
    N = len(list_imgs)
    fig, ax = plt.subplots(1, N, figsize=(N*size[0], size[1]))
    if N==1:
        img = list_imgs[0]
        if len(img.shape)==3:
            img = img[:, :, img.shape[2]//2]
        im = ax.imshow(img)
        plt.colorbar(im)
    else:
        for i,img in enumerate(list_imgs):
            if len(img.shape)==3:
                img = img[:, :, img.shape[2]//2]
            im = ax[i].imshow(img)
            plt.colorbar(im)