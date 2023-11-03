import os
import re
import subprocess
import torchio as tio
from ml.ControllerClass import Controller
from ml.SampleClass import Sample
from ml.models.HessNet import HessBlock, HessNet, HessNet2, GaussianBlur3D, HessianTorch
from ml.models.unet3d import U_Net, U_HessNet, ParallelNet
from ml.models.unet2d import U_Net2d
from scripts.load_and_save import save_vol_as_nii
from scripts.utils import get_path


class Runner:
    def __init__(self, settings):
        self.settings = settings
        self.device = settings['device']
        self.model_name = settings['model']
        self.controller_dict = {'device' : self.device,
                                'is2d' : settings.get('is2d', False)}
        self.get_model()
        self.controller = Controller(self.controller_dict)
        self.controller.load(path_to_checkpoint=f"{settings['path_to_pretrained_models']}/{self.model_name}_{settings['test_name']}")


    def get_model(self):
        if self.model_name == 'Unet3d_16ch':
            self.controller_dict.update({'model' : U_Net(channels=16)})
        if self.model_name == 'Unet2d_16ch':
            self.controller_dict.update({'model' : U_Net2d(channels=16)})
        elif self.model_name == 'HessNet':
            self.controller_dict.update({'model' : HessNet(start_scale=[0.8], device=self.device)})

    
    def run_sample(self, sample : Sample):
        subject = sample.get_subject()
        if self.settings.get('normalization'):
            if self.settings['normalization'] == 'Z-norm':
                subject = tio.transforms.ZNormalization()(subject)
        if self.controller_dict['is2d']:
            predict_settings = {
             "patch_shape" : (512, 512, 1),
             "overlap_shape" : (0, 0, 0),
             "batch_size" : 1,
             "num_workers": 4,
            }
        else:
            predict_settings = {
             "patch_shape" : (256, 256, 64),
             "overlap_shape" : (32, 32, 24),
             "batch_size" : 1,
             "num_workers": 4,
            }   
        seg = self.controller.single_predict(subject, predict_settings)
        path_to_save_sample = f"{sample.path_to_sample_dir}/{sample.sample_name}_segmentation_{self.model_name}.nii.gz"
        save_vol_as_nii(seg, subject.head.affine, path_to_save_sample)
        

    def get_metrics(self, sample: Sample, metrics, for_masked, save=False) -> str:
        metric_dict = {}
        if for_masked:
            path = sample.make_masked(mask_by='brain',
                                      keys_to_mask=['vessels',
                                                    f'segmentation_{self.model_name}'])
        else:
            path = sample.path_to_sample_dir
        
        
        GT_path = get_path(path, 'vessels')
        SEG_path = get_path(path, f'segmentation_{self.model_name}')
        command_output = subprocess.run([f"{self.settings['path_to_EvaluateSegmentation']}",
                                     GT_path, SEG_path], stdout=subprocess.PIPE, text=True)
        
        if save:
            if for_masked:
                text_file = open(f"{sample.path_to_sample_dir}/metrics_masked.txt", "w")
            else:
                text_file = open(f"{sample.path_to_sample_dir}/metrics.txt", "w")
            text_file.write(command_output.stdout)
            text_file.close()
        
        command_output = command_output.stdout.split('\n')
        for metric in metrics:
            for line in command_output:
                if re.search(metric, line):
                    metric_dict.update({metric : line.split('\t')[1][2:]})
        
        return(metric_dict)
    
    
    