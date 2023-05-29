from typing import Dict

import os
import torch
from numpy import asarray
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, RMSprop, Adamax, NAdam
from tqdm import tqdm



class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.n_epochs = config['n_epochs']
        self.optimizer = None
        self.opt_fn = lambda model: Adam(model.parameters(), lr=config['lr'])
        self.model = None
        self.history = None
        self.loss_fn = config["loss"]
        self.device = config['device']
        self.verbose = config.get('verbose', True)
        
        
    def fit(self, model, train_dataloader=None, val_dataloader=None):
        self.model = model.to(self.device)
        self.optimizer = self.opt_fn(model)
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }

        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}/{self.n_epochs}")
            train_info = self.train_epoch(train_dataloader)
            print(train_info)
            self.history['train_loss'].append(train_info['loss'])
            #val_info = self.val_epoch(val_dataloader)
            #self.history['val_loss'].extend([val_info['loss']])
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
            output = outputs[0]    
            
            loss = self.loss_fn(output, vessels_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_val = loss.item()
            losses.append(loss_val)
        return {'loss': sum(losses)/len(losses)}

    
    def val_epoch(self, val_dataloader):
        pass

    
    def predict(self, test_dataloader):
        pass

    
    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("You should train the model first")
        checkpoint = {
            "trainer_config": self.config,
            "model_state_dict": self.model.state_dict()
        }
        torch.save(checkpoint, path)

    
    @classmethod
    def load(cls, path: str):
        pass


def test_model(model, dataloader, device='cpu'):
    pass