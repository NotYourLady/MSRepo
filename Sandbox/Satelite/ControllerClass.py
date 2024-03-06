from typing import Dict
import os
import copy
from tqdm import tqdm
import torch
import torchio as tio


class Controller:
    def __init__(self, config: Dict):
        self.config = config
        self.device = config['device']
        self.verbose = config.get('verbose', False)
        
        self.epoch = 0
        self.model = config['model']
        self.model.to(self.device)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            "test_quality": [],
        }
        
        self.opt_fn = config.get('optimizer_fn', None)
        self.sheduler_fn = config.get('sheduler_fn', None)
        self.optimizer = None
        self.sheduler = None
        
        self.loss_fn = config.get('loss', None)
        self.metric_fn = config.get('metric', None)
        self.is2d = config.get('is2d', False)
        self.stop_test_cout = config.get('early_stopping', None)
        
    def fit(self, dataset, n_epochs):
        #if self.model is None:
        #    self.model = model.to(self.device)
        if self.optimizer is None:
            self.optimizer = self.opt_fn(self.model)
        if self.sheduler is None:
            if self.sheduler_fn is not None:
                self.sheduler = self.sheduler_fn(self.optimizer)
        
        start_epoch = self.epoch
        best_test_val = 0
        count_without_new_best_test_val = 0
        for epoch in range(start_epoch, start_epoch+n_epochs):
            self.epoch += 1
            print(f"Epoch {epoch + 1}/{start_epoch+n_epochs}")

            if dataset.train_dataloader is not None:
                train_info = self.train_epoch(dataset.train_dataloader)
                print(train_info)
                self.history['train_loss'].append(train_info['mean_loss'])
            
            if dataset.val_dataloader is not None:
                val_info = self.val_epoch(dataset.val_dataloader)
                print(val_info)
                self.history['val_loss'].append(val_info)
            
            if dataset.test_dataloader is not None:
                test_info = self.test_epoch(dataset.test_dataloader)
                self.history['test_quality'].append(test_info)

                test_val = 0
                for test in test_info:
                    print(test)
                    test_val+=test["metric"]
                test_val/=len(test_info)
                if test_val>best_test_val:
                    best_test_val=test_val
                    count_without_new_best_test_val=0
                    print('new best!')
                else:
                    count_without_new_best_test_val+=1
                    print('count_without_new_best_test_val:', count_without_new_best_test_val)

                if self.stop_test_cout:
                    if count_without_new_best_test_val>=self.stop_test_cout:
                        print("Early stopping!")
                        return self.model.eval()
                
            if self.sheduler is not None:
                self.sheduler.step()
            
        return self.model.eval()

    
    def train_epoch(self, train_dataloader):
        self.model.train()
        losses = []
        if self.verbose:
            train_dataloader = tqdm(train_dataloader)
        for patches_batch in train_dataloader:
            img_batch = patches_batch['img']['data'].float().to(self.device)  
            label_batch = patches_batch['label']['data'].float().to(self.device) 
            
            if self.is2d:               
                img_batch = img_batch[:, :, :, :, 0]
                label_batch = label_batch[:, :, :, :, 0]
            
            outputs = self.model.forward(img_batch)   
            loss = self.loss_fn(label_batch, outputs)

            self.optimizer.zero_grad()
            loss.backward()
            
            #loss.register_hook(lambda grad: print(grad))
            
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
            patch_loader = batch["patch_loader"]
            grid_aggregator = batch["grid_aggregator"]
            GT = batch["GT"]
            sample_name = batch["sample_name"]
            
            head_seg = self.fast_predict(patch_loader, grid_aggregator)
            metric = self.metric_fn(GT.data, head_seg)
            metrics.append({"sample" : sample_name,
                            #"seg_sum/GT_sum" : head_seg.sum()/GT.data.sum(),
                            "metric" : metric})
    
        return metrics
    

    def test_epoch(self, test_dataloader):
        self.model.eval()
        metrics = []
        if self.verbose:
            test_dataloader = tqdm(test_dataloader)
        for batch in test_dataloader:
            patch_loader = batch["patch_loader"]
            grid_aggregator = batch["grid_aggregator"]
            GT = batch["GT"]
            sample_name = batch["sample_name"]
            
            head_seg = self.fast_predict(patch_loader, grid_aggregator)
            metric = self.metric_fn(GT.data, head_seg)
            metrics.append({"sample" : sample_name,
                            #"seg_sum/GT_sum" : head_seg.sum()/GT.data.sum(),
                            "metric" : metric})
    
        return metrics
    
    
    def fast_predict(self, patch_loader, grid_aggregator, thresh=0.5):
        for patches_batch in patch_loader:
            patch_locations = patches_batch[tio.LOCATION]
            head_patches = patches_batch['img']['data'].to(self.device)
            if self.is2d:
                head_patches = head_patches[:, :, :, :, 0]
            with torch.no_grad():
                patch_seg = self.model(head_patches)
                if self.is2d:
                    patch_seg = patch_seg.unsqueeze(-1)
                grid_aggregator.add_batch(patch_seg.detach().cpu(), patch_locations)
        
        seg = grid_aggregator.get_output_tensor()
        if thresh is not None: 
            seg = torch.where(seg>thresh, 1, 0)
        return(seg)
    

    def predict(self, test_dataloader, path_to_save=None):
        self.model.eval()
        metrics = []
        if self.verbose:
            test_dataloader = tqdm(test_dataloader)
        for batch in test_dataloader:
            patch_loader = batch["patch_loader"]
            grid_aggregator = batch["grid_aggregator"]
            GT = batch["GT"]
            sample_name = batch["sample_name"]
            head_seg = self.fast_predict(patch_loader, grid_aggregator)
            metric = self.metric_fn(GT.data, head_seg)
            #print(GT.data.sum(), head_seg.sum())
            metric = {"sample" : sample_name,
                      "seg_sum/GT_sum" : head_seg.sum()/GT.data.sum(),
                      "metric1" : metric}
            metrics.append(metric)
            if path_to_save is not None:
                path_to_save_seg = path_to_save + '/' + sample_name + '.nii.gz'
                segImage = tio.Image(tensor=head_seg, affine=GT.affine)
                segImage.save(path_to_save_seg)
        return metrics
    
    
    def single_predict(self, subject, settings):
        grid_sampler = tio.GridSampler(subject,
                                       patch_size=settings["patch_shape"],
                                       patch_overlap=settings["overlap_shape"])
        grid_aggregator = tio.data.GridAggregator(sampler=grid_sampler, overlap_mode='hann')
        patch_loader = torch.utils.data.DataLoader(grid_sampler,
                                                   batch_size=settings["batch_size"],
                                                   num_workers=settings["num_workers"])
        seg = self.fast_predict(patch_loader, grid_aggregator)
        return(seg)
          
    
    def save_weights(self, path: str):
        if self.model is None:
            raise RuntimeError("Need a model")
        torch.save(self.model.state_dict(), path)
    
    
    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("Need a model")
        save_config = copy.deepcopy(self.config)
        if save_config.get('optimizer_fn'):
            del save_config['optimizer_fn']
        if save_config.get('sheduler_fn'):
            del save_config['sheduler_fn']
        checkpoint = {
            "trainer_config": save_config,
            "verbose" : self.verbose,

            "epoch" : self.epoch,
            "history" : self.history,

            "optimizer_state_dict" : self.optimizer.state_dict(),
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
        self.verbose = checkpoint["verbose"]

        self.epoch = checkpoint['epoch']
        self.history = checkpoint["history"]

        self.loss_fn = checkpoint["loss_fn"]
        self.metric_fn = checkpoint["metric_fn"]

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        if self.opt_fn:
            self.optimizer = self.opt_fn(self.model)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.sheduler = checkpoint["sheduler"]

    @classmethod
    def load_model(cls, model, path_to_checkpoint):
        checkpoint = torch.load(path_to_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])