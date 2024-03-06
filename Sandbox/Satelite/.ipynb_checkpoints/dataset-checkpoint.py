import os
import re
import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


def get_path(path, key="head"):
    out = []
    names = os.listdir(path)
    for name in names:
        m = re.search(key, name)
        if m:
            out.append(f"{path}/{name}")
    
    if len(out)==1:
        return(out[0])
    return(out)


class TioDataset(Dataset):
    def __init__(self, data_dir,
                 train_settings=None,
                 val_settings=None,
                 test_settings=None,
                 paths=None):
        
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.paths = paths
        self.train_settings = train_settings
        self.val_settings = val_settings
        self.test_settings = test_settings
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        
        if train_settings is not None:
            self.train_data = self.set_data(data_type='train')
            self.train_dataloader = self.set_dataloader(data_type='train')
            
        if val_settings is not None:
            self.val_data = self.set_data(data_type='val')
            self.val_dataloader = self.set_dataloader(data_type='val')
            
        if test_settings is not None:
            self.test_data = self.set_data(data_type='test')
            self.test_dataloader = self.set_dataloader(data_type='test')
            
        
    def set_data(self, data_type):
        subjects_list = []
        if data_type=='train':
            path_to_data = self.data_dir + "/train"
        elif data_type=='val':
            path_to_data = self.data_dir + "/val"
        elif data_type=='test':
            path_to_data = self.data_dir + "/test"
        else:
            raise RuntimeError("Dataset::set_data ERROR")

        for dirname, dirnames, filenames in os.walk(path_to_data):
            for subdirname in dirnames:
                if subdirname[0]!='.':
                    p = os.path.join(dirname, subdirname)
                    subject_dict = {"sample_name" : subdirname}
                    if get_path(p, 'img'):
                        subject_dict.update({'img': tio.ScalarImage(get_path(p, 'img'))})
                    if get_path(p, 'label'):
                        subject_dict.update({'label': tio.LabelMap(get_path(p, 'label'))})
                    subject = tio.Subject(subject_dict)
                    subjects_list.append(subject)
        
        return(tio.SubjectsDataset(subjects_list))


    def add_prob_map(self, subject, focus=1.5):
        _, h, w, d = subject.shape
        x0 = h//2
        y0 = w//2
        prob_slice = np.ones((h,w))

        for x in range(prob_slice.shape[0]):
            for y in range(prob_slice.shape[1]):
                prob_slice[x, y] = ((focus-((x/x0-1)**2 + (y/y0-1)**2)**0.5))

        prob_slice = prob_slice.clip(0, 1)
        prob_vol = np.stack(d*[prob_slice,], axis=2)

        prob_Image = tio.Image(tensor=torch.tensor(prob_vol).unsqueeze(0),
                               type=tio.SAMPLING_MAP,
                               affine=subject.head.affine)
        subject.add_image(prob_Image, "prob_map")
        return(subject)
    
    
    def set_dataloader(self, data_type):
        if data_type=='train':
            settings = self.train_settings
            data = self.train_data
        elif data_type=='val':
            settings = self.val_settings
            data = self.val_data
        elif data_type=='test':
            settings = self.test_settings
            data = self.test_data
        else:
            raise RuntimeError("Dataset::set_data ERROR")    
        
        
        if data_type=='train': 
            sampler = tio.data.UniformSampler(settings["patch_shape"])
            patches_queue = tio.Queue(
                data,
                settings["patches_queue_length"],
                settings["patches_per_volume"],
                sampler,
                num_workers=settings["num_workers"],
            )
            patches_loader = DataLoader(
                patches_queue,
                batch_size=settings["batch_size"],
                num_workers=0,  #must be
            )
            return(patches_loader)
        elif data_type in ('test', 'val'):
            test_loaders = []
            for subject in data:
                grid_sampler = tio.GridSampler(subject,
                                               patch_size=settings["patch_shape"],
                                               patch_overlap=settings["overlap_shape"])
                grid_aggregator = tio.data.GridAggregator(sampler=grid_sampler, 
                                                          overlap_mode='hann')
                patch_loader = torch.utils.data.DataLoader(grid_sampler,
                                                           batch_size=settings["batch_size"],
                                                           num_workers=settings["num_workers"])
                if ("label" in subject.keys()):
                    GT = subject.label
                else:
                    raise RuntimeError("Dataset::set_data ERROR: label in subject.keys()")    
                sample_name = subject.sample_name
                test_loaders.append({"patch_loader" : patch_loader,
                                     "grid_aggregator" : grid_aggregator,
                                     "GT" : GT,
                                     "sample_name" : sample_name})
            return(test_loaders)