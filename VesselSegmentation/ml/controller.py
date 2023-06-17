from typing import Dict
import os
import copy
from numpy import asarray
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
import torchio as tio

class Controller:
    def __init__(self, config: Dict):
        self.config = config
        self.device = config['device']
        self.verbose = config.get('verbose', True)
        
        self.epoch = 0
        self.model = None
        self.history = None
        
        self.opt_fn = config['optimizer_fn']
        self.sheduler_fn = config['sheduler_fn']
        self.optimizer = None
        self.sheduler = None
        
        self.loss_fn = config["loss"]
        self.metric_fn = config["metric"]
        
        
    def fit(self, model, dataset, n_epochs):
        self.model = model.to(self.device)
        self.optimizer = self.opt_fn(self.model)
        if self.sheduler_fn is not None:
            self.sheduler = self.sheduler_fn(self.optimizer)
        
        self.history = {
            'train_loss': [],
            'val_quality': [],
        }
        start_epoch = self.epoch
        for epoch in range(start_epoch, start_epoch+n_epochs):
            self.epoch += 1
            print(f"Epoch {epoch + 1}/{start_epoch+n_epochs}")
            
            train_info = self.train_epoch(dataset.train_dataloader)
            print(train_info)
            self.history['train_loss'].append(train_info['mean_loss'])
            
            val_info = self.val_epoch(dataset.test_dataloader)
            print(val_info)
            self.history['val_quality'].append(val_info['metrics'])
            
            if self.sheduler is not None:
                self.sheduler.step()
            
        return self.model.eval()

    
    def train_epoch(self, train_dataloader):
        self.model.train()
        losses = []
        if self.verbose:
            train_dataloader = tqdm(train_dataloader)
        for patches_batch in train_dataloader:
            head_batch = patches_batch['head']['data'].float().to(self.device)  
            vessels_batch = patches_batch['vessels']['data'].float().to(self.device) 
            
            #outputs = self.model.forward(head_batch)   
            outputs = self.model.forward(head_batch)[0]   
            loss = self.loss_fn(vessels_batch, outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_val = loss.item()
            losses.append(loss_val)
        return {'mean_loss': sum(losses)/len(losses)}
    

    def val_epoch(self, val_dataloader):
        self.model.eval()
        metrics = []
        if self.verbose:
            val_dataloader = tqdm(val_dataloader)
        for batch in val_dataloader:
            patch_loader, grid_aggregator, GT, sample_name = batch
            head_seg = self.fast_predict(patch_loader, grid_aggregator)
            metric = self.metric_fn(GT.data, head_seg)
            metrics.append({"sample" : sample_name,
                            "seg_sum/GT_sum" : head_seg.sum()/GT.data.sum(),
                            "metric1" : metric})
    
        return {'metrics': metrics}
    
    
    def fast_predict(self, patch_loader, grid_aggregator, thresh=0.5):
        for patches_batch in patch_loader:
            patch_locations = patches_batch[tio.LOCATION]
            head_patches = patches_batch['head']['data'].to(self.device)
            with torch.no_grad():
                grid_aggregator.add_batch(self.model(head_patches)[0].cpu(), patch_locations)
        seg = grid_aggregator.get_output_tensor()
        seg[seg<thresh] = 0
        seg[seg>0] = 1
        return(seg)


    def predict(self, test_dataloader, path_to_save=None):
        self.model.eval()
        metrics = []
        if self.verbose:
            test_dataloader = tqdm(test_dataloader)
        for batch in test_dataloader:
            patch_loader, grid_aggregator, GT, sample_name = batch
            head_seg = self.fast_predict(patch_loader, grid_aggregator)
            metric = self.metric_fn(GT.data, head_seg)
            metric = self.metric_fn(GT.data, head_seg)
            metric = {"sample" : sample_name,
                      "seg_sum/GT_sum" : head_seg.sum()/GT.data.sum(),
                      "metric1" : metric}
            print(metric)
            metrics.append(metric)
            if path_to_save is not None:
                path_to_save_seg = path_to_save + '/' + sample_name + '.nii.gz'
                segImage = tio.Image(tensor=head_seg, affine=GT.affine)
                segImage.save(path_to_save_seg)
                
        return metrics
    
    
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

            "epoch" : self.epoch,
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

        self.epoch = checkpoint['epoch']
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
