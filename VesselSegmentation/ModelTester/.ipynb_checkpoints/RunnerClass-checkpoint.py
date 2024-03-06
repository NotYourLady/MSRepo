import os
import re
import json
import subprocess
from tqdm import tqdm
import torchio as tio


from ml.get_model import get_model
from ml.ControllerClass import Controller
from ml.SampleClass import Sample
from scripts.load_and_save import save_vol_as_nii
from scripts.utils import get_path


class Runner:
    def __init__(self, settings, test):
        self.test = test
        self.settings = settings
        self.learning_settings = settings['learning_settings']
        self.device = self.learning_settings['device']
        self.model_name = settings['model']
        self.controller = self.init_controller()
        self.test_paths = settings['tests'][test]['test']

        self.avg_metrics = {}
        self.avg_metrics_mask = {}

        try:
            if not os.path.exists(self.settings['results_path']):
                os.mkdir(self.settings['results_path'])
        except:
            raise RuntimeError("Runner::__init__: invalid results_path!")
    
    def init_controller(self):
        model = get_model(self.model_name)
        controller_config = {
            'device' : self.device,
            "model" : model,
            "is2d" : self.learning_settings['is2d']
        }
        controller = Controller(controller_config)
        controller.load(path_to_checkpoint = f"{self.settings['trained_models_path']}/" + \
                                             f"{self.model_name}_{self.test}")
        return controller


    def run(self):
        for m in self.settings['metrics']:
            self.avg_metrics.update({m : []})
            self.avg_metrics_mask.update({m : []})
        
        
        for path in tqdm(self.test_paths):
            sample = Sample(path)
            self.run_sample(sample, save_masked=True)
            
            metrics = self.get_metrics(sample, metrics=self.settings['metrics'],
                                         save=False, for_masked=False)
            metrics_masked = self.get_metrics(sample, metrics=self.settings['metrics'],
                                                save=False, for_masked=True)
            print("metrics:", metrics)
            print("metrics_masked:", metrics_masked)
            for m in self.settings['metrics']:
                self.avg_metrics[m].append(metrics[m])
                self.avg_metrics_mask[m].append(metrics_masked[m])
        for m in self.settings['metrics']:
            self.avg_metrics[m] = sum([float(x) for x in self.avg_metrics[m]])/len(self.avg_metrics[m])
            self.avg_metrics_mask[m] = sum([float(x) for x in self.avg_metrics_mask[m]])/len(self.avg_metrics_mask[m])
    
        print("avg_metrics:", self.avg_metrics)
        print("avg_metrics_mask:", self.avg_metrics_mask)
        self.dump_metrics()

    
    def run_sample(self, sample : Sample, save_masked=False):
        subject = sample.get_subject()
        if self.settings.get('normalization'):
            if self.settings['normalization'] == 'Z-norm':
                subject = tio.transforms.ZNormalization()(subject)
        
        BATCH_SIZE_TEST = self.learning_settings['test_batch_size']
        PATCH_SIZE_TEST = self.learning_settings['patch_size_test']
        OVERLAP_TEST = self.learning_settings['overlap_test']
        NUM_WORKERS = self.learning_settings['num_workers']
        
        predict_settings = {
            "patch_shape" : PATCH_SIZE_TEST,
            "overlap_shape" : OVERLAP_TEST,
            "batch_size" : BATCH_SIZE_TEST,
            "num_workers": NUM_WORKERS,
        }
        
        seg = self.controller.single_predict(subject, predict_settings)
        path_out = self.settings['segmentation_out_path']
        path_to_save_seg = f"{path_out}/{sample.sample_name}_seg_{self.model_name}.nii.gz"
        save_vol_as_nii(seg, subject.head.affine, path_to_save_seg)
        if save_masked:
            mask = subject.brain.data
            masked_gt = subject.vessels.data * mask
            masked_seg = seg * mask
            path_to_save_seg_masked = f"{path_out}/{sample.sample_name}_seg_masked_{self.model_name}.nii.gz"
            path_to_save_gt_masked = f"{path_out}/{sample.sample_name}_gt_masked_{self.model_name}.nii.gz"
            save_vol_as_nii(masked_seg, subject.head.affine, path_to_save_seg_masked)
            save_vol_as_nii(masked_gt, subject.head.affine, path_to_save_gt_masked)

    
    def get_metrics(self, sample: Sample, metrics, for_masked, save=False, save_path=None) -> str:
        metric_dict = {}

        path = self.settings['segmentation_out_path']
        if for_masked:
            GT_path = f"{path}/{sample.sample_name}_seg_masked_{self.model_name}.nii.gz"
            SEG_path = f"{path}/{sample.sample_name}_gt_masked_{self.model_name}.nii.gz"
        else:
            GT_path = get_path(sample.path_to_sample_dir, 'vessels')
            SEG_path = f"{path}/{sample.sample_name}_seg_{self.model_name}.nii.gz"
        
        command_output = subprocess.run([f"{self.settings['path_to_EvaluateSegmentation']}",
                                     GT_path, SEG_path], stdout=subprocess.PIPE, text=True)
        
        if save:
            if for_masked:
                text_file = open(f"{save_path}/metrics_masked.txt", "w")
            else:
                text_file = open(f"{save_path}/metrics.txt", "w")
            text_file.write(command_output.stdout)
            text_file.close()
        
        command_output = command_output.stdout.split('\n')
        for metric in metrics:
            for line in command_output:
                if re.search(metric, line):
                    metric_dict.update({metric : line.split('\t')[1][2:]})
        
        return(metric_dict)

    
    def dump_metrics(self):
        test_dict = {
                self.settings['model']: {
                    self.test: {
                        "avg_metrics" : self.avg_metrics,
                        "avg_metrics_mask" : self.avg_metrics_mask
                    }   
                }
            }

        path_to_out = f"{self.settings['results_path']}/avg_metrics.json"
                
        if os.path.exists(path_to_out):
            fp = open(path_to_out, 'r+')
            prev_dict = json.load(fp)
            if prev_dict.get(self.settings['model']):
                prev_dict[self.settings['model']].update(test_dict[self.settings['model']])
            else:
                prev_dict.update(test_dict)
            fp.seek(0)
            json.dump(prev_dict, sort_keys=True, indent=4, fp=fp)
            fp.truncate() 
        else:
            fp = open(path_to_out, 'w')
            json.dump(test_dict, sort_keys=True, indent=4, fp=fp)
            fp.close() 
    