import psutil
import numpy as np
import os
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import skimage.io
import pandas as pd
import torchvision
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def get_noise(size, device, noise_coef=1.0e-2):
    noise =  noise_coef * torch.randn(*size, device=device)
    return(noise)    


def get_noised_labels(size, type_, device, noise_coef=1.0e-2):
    assert type_!=1 or type_!=0
    noise = torch.abs(get_noise(size, noise_coef))
    if type_==1:
        labels = torch.ones(size, device=device)
        labels -= noise
        return(labels)
    if type_==0:
        labels = torch.zeros(size, device=device)
        labels += noise
        return(labels)

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def raise_edges(img, l=5, h=50, t="uint"):
    assert (t!="uint" or t!="float")
    if(t=="uint"):
        g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(g_img,l,h,L2gradient=True)
        img[edges.astype(bool), :] = 0
    elif(t=="float"):
        img = (255*img).astype(np.uint8)
        g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(g_img,l,h,L2gradient=True)
        img[edges.astype(bool), :] = 0
        img = img.astype(np.float32)/255
    return(img)

def smart_resize(img, size=256):
    if len(img.shape)==2:
        img=np.array(3*[img,])
        img = np.transpose(img, (1, 2, 0))

    max_sh = np.argmax(img.shape[:2])
    min_sh = np.argmin(img.shape[:2])
    max_sz = img.shape[max_sh]
    min_sz = img.shape[min_sh]
    delta = max_sz - min_sz
    new_img = 255*np.ones((max_sz, max_sz, 3)).astype(np.uint8)
    if (max_sh == 1):
        new_img[(delta//2):(min_sz+delta//2), :] = img
    else:
        new_img[:, (delta//2):(min_sz+delta//2)] = img
    new_img = resize(new_img,[size,size])
    return(new_img)


hflipper = T.RandomHorizontalFlip(p=0.5)
jitter = T.ColorJitter(brightness=0, hue=0.4)
affiner = T.RandomAffine(degrees=(-60, 60), translate=(0.05, 0.01), scale=(0.8, 0.9), fill=1)
grayer = T.Grayscale(num_output_channels=1)


def augmentation(img, n=4, to_gray=False, with_sharpness=None):
    img = torch.from_numpy(img.transpose(2,0,1)).to(dtype=torch.float)
    
    transforms=[hflipper, affiner]
    if to_gray:
        transforms.append(grayer)
    applier = T.RandomApply(transforms=transforms,p=1 )
    
    if with_sharpness is not None:
        img = T.functional.adjust_sharpness(img, sharpness_factor=5) 

    imgs = [applier(img) for _ in range(n)]
    return(imgs)


def build_pokemon_dataset(data_dirs=["images"], img_size=256,
                          aug_size=3, to_gray=False,
                          edges=None, with_sharpness=None,
                          memSTOP = 2):
    GiG = 2**30
    pokemon_dataset = []
    count = 0
    for data_dir in data_dirs:
        for dirpath, dirnames, filenames in tqdm(os.walk(data_dir)):
            for fname in filenames:
                if fname.endswith(".jpg"):
                    pokemon = skimage.io.imread(dirpath + "/" + fname)
                    pokemon = smart_resize(pokemon, size=img_size)
                    if edges is not None:
                        pokemon = raise_edges(pokemon, edges[0], edges[1], "float")
                    aug_pokemon = augmentation(pokemon, aug_size,
                                               to_gray=to_gray,
                                               with_sharpness=with_sharpness)
                    pokemon_dataset+=aug_pokemon
                    count += 1
            #if count>10:
            if (psutil.virtual_memory().available/(GiG) < memSTOP):   
                print(f"Only {memSTOP}GB system memory left :(")         
                random.shuffle(pokemon_dataset)
                pokemon_dataset = torch.stack(pokemon_dataset)
                pokemon_dataset = T.Normalize(*stats)(pokemon_dataset)
                return pokemon_dataset
    
    random.shuffle(pokemon_dataset)
    pokemon_dataset = torch.stack(pokemon_dataset)
    pokemon_dataset = T.Normalize(*stats)(pokemon_dataset)
    return pokemon_dataset      


def t2i(tens, indx=(1, 2, 0)):
    tens = torch.FloatTensor(tens)
    tens = tens.permute(indx[0], indx[1], indx[2])
    return(tens.numpy())    


def show_intermediate_results(autoencoder, val_loader, device, n=3):
    with torch.no_grad():
        for batch in val_loader:
            out = autoencoder(batch.to(device))
            if(type(out) is tuple):
                reconstruction, _,_ = out
            else:
                reconstruction = out    
            result = reconstruction.cpu().detach().numpy()
            ground_truth = batch.numpy()
            break      
    plt.figure(figsize=(6*n, 2))
    for i, (gt, res) in enumerate(zip(ground_truth[:n], result[:n])):
        plt.subplot(1, 2*n, 2*i+1)
        gt = t2i(gt)
        plt.imshow(gt)
        plt.subplot(1, 2*n, 2*i+2)
        res = t2i(res)
        plt.imshow(res)         
    plt.show()


def save_samples(gen, index, latent_tensors,sample_dir, show=True):
    gen.eval()
    fake_images = gen(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        nmax=64
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(denorm(fake_images.cpu().detach())[:nmax], nrow=8).permute(1, 2, 0))
        plt.show()
stats = (0.5, 0.5)


def denorm(img_tensors):
    return img_tensors * stats[0] + stats[1]


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))


def show_batch(dl, nmax=64):
    for images in dl:
        show_images(images, nmax)
        break
        

def save_model(model, path_to_save_model, epoch):
    path_to_save_model_ = path_to_save_model + f"/epoch_{epoch}_{int(time.time()//60)-27963162}" 
    os.makedirs(path_to_save_model_, exist_ok=True)
    torch.save(model['discriminator'].state_dict(), path_to_save_model_ + "/discriminator")
    torch.save(model['generator'].state_dict(), path_to_save_model_ + "/generator")        

