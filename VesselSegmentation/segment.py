import os
import torchio as tio
import json
import sys
import argparse

from ml.controller import Controller
from ml.models.unet_deepsup import U_Net_DeepSup
from ml.models.activates import swish


def segment(settings):
    controller_config = {'device' : settings["device"]}
    controller = Controller(controller_config)

    model = U_Net_DeepSup(channel_coef=32, act_fn=swish())
    model_name = "UnetMSS32_ExpLog09_34"
    path_to_check= "/home/msst/repo/MSRepo/VesselSegmentation/saved_models/" + model_name
    controller.load(model, path_to_checkpoint=path_to_check)
    
    controller.easy_predict(settings)


def load_config(path):
    with open(path, 'r') as openfile:
        config = json.load(openfile)
    return config


def main():
    parser = argparse.ArgumentParser(description='Videos to images')
    parser.add_argument('-c', '--config', type=str, help='path_to_config')
    args = parser.parse_args()
    
    settings = load_config(args.config)
    print(settings)
    segment(settings)
        
if __name__ == '__main__':
    main()  # next section explains the use of sys.exit