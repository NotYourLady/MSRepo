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
        print(self.device)
        self.verbose = config.get('verbose', True)
        
        self.epoch = 0
        self.model = None
        self.history = None
        
        self.opt_fn = config.get('optimizer_fn', None)
        self.sheduler_fn = config.get('sheduler_fn', None)
        self.optimizer = None
        self.sheduler = None
        
        self.loss_fn = config.get('loss', None)
        self.metric_fn = config.get('metric', None)
        
        
    def fit(self, model, dataset, n_epochs):
        if self.model is None:
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
        start_epoch = self.epoch
        for epoch in range(start_epoch, start_epoch+n_epochs):
            self.epoch += 1
            print(f"Epoch {epoch + 1}/{start_epoch+n_epochs}")
            
            train_info = self.train_epoch(dataset.train_dataloader)
            print(train_info)
            self.history['train_loss'].append(train_info['mean_loss'])
            
            val_info = self.val_epoch(dataset.val_dataloader)
            print(val_info)
            self.history['val_loss'].append(val_info['mean_loss'])
            
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
        for patches_batch in train_dataloader:
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
        return {'mean_loss': sum(losses)/len(losses)}
    
    
    def val_epoch(self, val_dataloader):
        self.model.eval()
        losses = []
        if self.verbose:
            val_dataloader = tqdm(val_dataloader)
        for patches_batch in val_dataloader: 
            head_batch = patches_batch['head']['data'].float().to(self.device)  
            vessels_batch = patches_batch['vessels']['data'].float().to(self.device) 
            with torch.no_grad():
                outputs = self.model.forward(head_batch)   
                loss = self.loss_fn(vessels_batch, outputs)
                loss_val = loss.item()
                losses.append(loss_val)
        return {'mean_loss': sum(losses)/len(losses)}
    

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
                            "seg_sum/GT_sum" : head_seg.sum()/GT.data.sum(),
                            "metric1" : metric})
    
        return {'metrics': metrics}
    
    
    def fast_predict(self, patch_loader, grid_aggregator, thresh=0.5):
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
    
    
    ### Settings Example:
    # settings = {"path_in": "/home/user/head.nii.gz",
    #         "path_out": "/home/user/segmented_head.nii.gz",
    #         "nn_settings":{
    #             "patch_shape" : (256, 256, 128),
    #             "overlap_shape" : (32, 32, 24),
    #             "batch_size" : 1,
    #             "num_workers": 4,
    #             }
    #         }
    ###
    def easy_predict(self, settings):
        sample_name = os.path.basename(settings['path_in'])
        subject_dict = {'head': tio.ScalarImage(settings['path_in'])}
        subject = tio.Subject(subject_dict)
        transforms = [tio.transforms.Resample(target=0.5),
                      tio.transforms.ZNormalization()]
        transform = tio.Compose(transforms)
        subject = transform(subject)

        nn_settings = settings['nn_settings']
        grid_sampler = tio.GridSampler(subject,
                                       patch_size=nn_settings["patch_shape"],
                                       patch_overlap=nn_settings["overlap_shape"])
        grid_aggregator = tio.data.GridAggregator(sampler=grid_sampler, overlap_mode='hann')
        patch_loader = torch.utils.data.DataLoader(grid_sampler,
                                                   batch_size=nn_settings["batch_size"],
                                                   num_workers=nn_settings["num_workers"])
        seg = self.fast_predict(patch_loader, grid_aggregator)
        
        segImage = tio.LabelMap(tensor=seg, affine=subject.head.affine)
        reverse_transform = tio.transforms.Resample(target=settings['path_in'])
        segImage = reverse_transform(segImage)
        segImage.save(settings["path_out"])
        
        
        
    
    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("Need a model")
        save_config = copy.deepcopy(self.config)
        del save_config['optimizer_fn']
        del save_config['sheduler_fn']
        checkpoint = {
            "trainer_config": save_config,
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
        self.verbose = checkpoint["verbose"]

        self.epoch = checkpoint['epoch']
        self.history = checkpoint["history"]

        self.optimizer = checkpoint["optimizer"]
        self.sheduler = checkpoint["sheduler"]

        self.loss_fn = checkpoint["loss_fn"]
        self.metric_fn = checkpoint["metric_fn"]

        self.model = model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)


    @classmethod
    def load_model(cls, model, path_to_checkpoint):
        checkpoint = torch.load(path_to_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
