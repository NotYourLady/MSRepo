import os
import torch
import torch.nn as nn

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
    
    
def test_model(model, device="cuda", n_epochs=100, input_shape=(1,1,64,64,64)):
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.rand(*input_shape)
    GT = torch.rand(*input_shape)
    
    model = model.to(device)
    for epoch in range(n_epochs):
        x_device = x.to(device)
        GT_device = GT.to(device)

        out = model.forward(x_device)   
        loss = loss_fn(GT_device, out[0])

        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch%10==0:
            print(loss.item())
    