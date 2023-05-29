import os
import torch

def get_total_params(model):
    total_params = sum(
    param.numel() for param in model.parameters()
    )
    return(total_params)


def save_model(model, path_to_save_model, name):
    os.makedirs(path_to_save_model, exist_ok=True)
    torch.save(model.state_dict(), path_to_save_model + "/" + name) 
    
def load_pretrainned(model, path_to_weights):
    model.load_state_dict(torch.load(path_to_weights))
    