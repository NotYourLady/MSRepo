from ml.models.HessNet import HessNet, HessUNet, HessUNet2 
from ml.models.unet3d import U_Net, Unet3d
from ml.models.unet2d import U_Net2d
#from ml.models.UNETR import UNETR


def get_model(model_name, device='cuda'):
    if model_name == 'Unet3d_16ch':
        #return(Unet3d(channels=16))
        return(U_Net(channels=16))
        
    # elif model_name == 'UNETR':
    #     return(UNETR(in_channels=1, out_channels=1, img_size=(96, 96, 64),
    #           feature_size=16, hidden_size=512,
    #           mlp_dim=512, num_heads=4,
    #           norm_name='batch'))
        
    elif model_name == 'Unet2d_16ch':
        return(U_Net2d(channels=16))
        
    elif model_name == 'HessUNet':
        return(HessUNet(in_channels=1, out_channels=1, channels=16, depth=3))
        
    elif model_name == 'HessUNet2':
        return(HessUNet2(in_channels=1, out_channels=1, channels=16, depth=3))
        
    elif model_name == 'HessNet':
        return(HessNet(start_scale=[0.8, 0.8, 1.2], device=device))