from typing import Dict

import os
import torch
import torchio as tio
from numpy import asarray
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, RMSprop, Adamax, NAdam
from tqdm import tqdm
import copy


from IPython.display import clear_output
import matplotlib.pyplot as plt
SHOW_SLICE = 32
def print_img(vol, axis, title= 'title', show_slice=None, cmap='hot'):
    global SHOW_SLICE
    if title is not None:
        axis.set_title(title)
    if show_slice is None:
        im = axis.imshow(vol[:, :, SHOW_SLICE], cmap=cmap)
    else: 
        im = axis.imshow(vol[:, :, show_slice], cmap=cmap)
    plt.colorbar(im)


class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = config['device']
        self.verbose = config.get('verbose', True)
        
        self.n_epochs = config['n_epochs']
        self.model = None
        self.history = None
        
        self.opt_fn = config['optimizer_fn']
        self.sheduler_fn = config['sheduler_fn']
        self.optimizer = None
        self.sheduler = None
        
        self.loss_fn = config["loss"]
        self.metric_fn = config["metric"]
        
        
    def fit(self, model, train_dataloader=None, val_dataloader=None, test_dataloader=None):
        self.model = model.to(self.device)
        if self.opt_fn is not None:
            self.optimizer = self.opt_fn(self.model)
        if self.sheduler_fn is not None:
            self.sheduler = self.sheduler_fn(self.optimizer)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            "test_quality": [],
        }

        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}/{self.n_epochs}")
            
            train_info = self.train_epoch(train_dataloader)
            #train_info = self.train_epoch_tio(train_dataloader)
            print(train_info)
            self.history['train_loss'].append(train_info['loss'])
            
            #val_info = self.val_epoch(val_dataloader)
            #print(val_info)
            #self.history['val_quality'].append(val_info['metrics'])
            
            test_info = self.test_epoch(dataset.test_dataloader)
            print(test_info)
            self.history['test_quality'].append(test_info['metrics'])
            
            
            if self.sheduler is not None:
                self.sheduler.step()
            
        return self.model.eval()

    
    def train_epoch(self, train_dataloader):
        self.model.train()
        losses = []
        if self.verbose:
            train_dataloader = tqdm(train_dataloader)
        for batch in train_dataloader:
            head_batch = batch['head_patch'].to(self.device)
            vessels_batch = batch['vessels_patch'].to(self.device)
            
            outputs = self.model.forward(head_batch)   
            
            loss = self.loss_fn(vessels_batch, outputs)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_val = loss.item()
            losses.append(loss_val)
        return {'loss': sum(losses)/len(losses)}
    
    
    def train_epoch_tio(self, train_dataloader):
        self.model.train()
        losses = []
        if self.verbose:
            train_dataloader = tqdm(train_dataloader)
        for patches_batch in train_dataloader:
            #head_batch = patches_batch['head']['data'].float().to(self.device)  
            #vessels_batch = patches_batch['vessels']['data'].float().to(self.device) 
            head_batch = patches_batch['head']['data'].float().to(self.device)  
            vessels_batch = patches_batch['vessels']['data'].float().to(self.device) 
        
            outputs = self.model.forward(head_batch)   
            #outputs = self.model.forward(head_batch)[0]   
            loss = self.loss_fn(vessels_batch, outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_val = loss.item()
            losses.append(loss_val)
        return {'loss': sum(losses)/len(losses)}

    
    def test_epoch(self, test_dataloader):
        self.model.eval()
        metrics = []
        if self.verbose:
            test_dataloader = tqdm(test_dataloader)
        for batch in test_dataloader:
            #patch_loader, grid_aggregator, GT, sample_name = batch
            patch_loader = batch["patch_loader"]
            grid_aggregator = batch["grid_aggregator"]
            GT = batch["GT"]
            sample_name = batch["sample_name"]
            
            head_seg = self.fast_predict(patch_loader, grid_aggregator)
            metric = self.metric_fn(GT.data, head_seg)
            metrics.append({"sample" : sample_name,
                            "seg_sum/GT_sum" : head_seg.sum()/GT.data.sum(),
                            "metric1" : metric})
    
        return {'metrics': metrics}
    
    
    def fast_predict(self, patch_loader, grid_aggregator, thresh=0.1):
        for patches_batch in patch_loader:
            patch_locations = patches_batch[tio.LOCATION]
            head_patches = patches_batch['head']['data'].to(self.device)
            with torch.no_grad():
                patch_seg = self.model(head_patches)
                grid_aggregator.add_batch(patch_seg[0].cpu(), patch_locations)
        seg = grid_aggregator.get_output_tensor()
        seg[seg<thresh]=0
        seg[seg>0]=1
        return(seg)
    
    
    
    def val_epoch(self, val_dataloader):
        patch_shape = val_dataloader.dataset.patch_shape
        self.model.eval()
        metrics = []
        sums = []
        if self.verbose:
            val_dataloader = tqdm(val_dataloader)
        for batch in val_dataloader:
            head_batch = batch['head']
            vessels_batch = batch['vessels']
            #brain_batch = batch['brain']
            head_seg = self.predict(head_batch, patch_shape)
            
            metric = self.metric_fn(vessels_batch, head_seg)
            metrics.append(metric)
            sums.append({"GT_sum" : vessels_batch.sum(),
                         "seg_sum" : head_seg.sum()})
            #return {'metrics': metric, "sums" : sums}
    
        return {'metrics': metrics, "sums" : sums}
    
    
    def predict(self, head_tensor_5_dim, ps, thresh=0.5):
        vol_shape = head_tensor_5_dim.shape
        s1 = vol_shape[2]//ps[0]#+1
        s2 = vol_shape[3]//ps[1]#+1
        s3 = vol_shape[4]//ps[2]#+1

        seg = torch.zeros_like(head_tensor_5_dim)
        with torch.no_grad():
            for i in range(s1):
                for j in range(s2):
                    for k in range(s3):
                        patch = head_tensor_5_dim[:,
                                                  :,
                                                  i*ps[0]:(i+1)*ps[0],
                                                  j*ps[1]:(j+1)*ps[1],
                                                  k*ps[2]:(k+1)*ps[2]].to(self.device)
                        seg[:,
                            :,
                            i*ps[0]:(i+1)*ps[0],
                            j*ps[1]:(j+1)*ps[1],
                            k*ps[2]:(k+1)*ps[2]] = self.model(patch)[0].cpu()

        seg[seg<thresh] = 0
        seg[seg>0] = 1
        return(seg)

    
    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("Need a model")
        save_config = copy.deepcopy(self.config)
        del save_config['optimizer_fn']
        del save_config['sheduler_fn']
        checkpoint = {
            "trainer_config": save_config,
            "device" : self.device,
            "verbose" : self.verbose,

            #"epoch" : self.epoch,
            "history" : self.history,

            "optimizer" : self.optimizer,
            "sheduler" : self.sheduler,

            "loss_fn" : self.loss_fn,
            "metric_fn" : self.metric_fn,
            
            "model_state_dict": self.model.state_dict()
        }
        torch.save(checkpoint, path)


    def load(self, model=None, path_to_checkpoint=None):
        if (self.model is None) and (model is None):
            raise RuntimeError("Need a model")
        checkpoint = torch.load(path_to_checkpoint)
        
        self.config = checkpoint["trainer_config"]
        self.device = checkpoint["device"]
        self.verbose = checkpoint["verbose"]

        #self.epoch = checkpoint['epoch']
        self.history = checkpoint["history"]

        self.optimizer = checkpoint["optimizer"]
        self.sheduler = checkpoint["sheduler"]

        self.loss_fn = checkpoint["loss_fn"]
        self.metric_fn = checkpoint["metric_fn"]

        self.model = model
        self.model.load_state_dict(checkpoint["model_state_dict"])


    @classmethod
    def load_model(cls, model, path_to_checkpoint):
        checkpoint = torch.load(path_to_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])