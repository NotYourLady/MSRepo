from typing import Dict

import os
import torch
from numpy import asarray
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, RMSprop, Adamax, NAdam
from tqdm import tqdm
import copy


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
        
        
    def fit(self, model, train_dataloader=None, val_dataloader=None):
        self.model = model.to(self.device)
        if self.opt_fn is not None:
            self.optimizer = self.opt_fn(self.model)
        if self.sheduler_fn is not None:
            self.sheduler = self.sheduler_fn(self.optimizer)
        
        self.history = {
            'train_loss': [],
            'val_quality': [],
        }

        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}/{self.n_epochs}")
            
            train_info = self.train_epoch(train_dataloader)
            print(train_info)
            self.history['train_loss'].append(train_info['loss'])
            
            val_info = self.val_epoch(val_dataloader)
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
        for batch in train_dataloader:
            head_batch = batch['head_patch'].to(self.device)
            vessels_batch = batch['vessels_patch'].to(self.device)
            #brain_batch = batch['brain_patch'].to(self.device)
            #print(head_batch.sum(), head_batch.min(), head_batch.max(), head_batch.mean())
            
            outputs = self.model.forward(head_batch)   
            
            loss = self.loss_fn(vessels_batch, outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_val = loss.item()
            losses.append(loss_val)
        return {'loss': sum(losses)/len(losses)}
    

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
            raise RuntimeError("You should train the model first")
        save_config = copy.deepcopy(self.config)
        del save_config['optimizer_fn']
        del save_config['sheduler_fn']
        checkpoint = {
            "trainer_config": save_config,
            "model_state_dict": self.model.state_dict()
        }
        torch.save(checkpoint, path)

    
    @classmethod
    def load(cls, path: str):
        pass


def test_model(model, dataloader, device='cpu'):
    pass