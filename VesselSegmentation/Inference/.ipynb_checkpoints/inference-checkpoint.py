import os
import json
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname('../.'))
from RunnerClass import Runner
from ml.SampleClass import Sample
from scripts.utils import get_path


def main(settings):
    runner = Runner(settings)
    path_to_data = settings['path_to_data']
    avg_metrics = {}
    avg_metrics_mask = {}
    for m in settings['metrics']:
        avg_metrics.update({m : []})
        avg_metrics_mask.update({m : []})
    
    fp = open(f"{settings['path_to_data']}/testing_info.json")
    test_settings = json.load(fp)
    fp.close()
    sample_names = test_settings[settings['test_name']]['test']
        
    for sample_name in tqdm(sample_names):
        print(f"process {sample_name}...")
        sample = Sample(f"{path_to_data}/{sample_name}")
        runner.run_sample(sample)
        metrics = runner.get_metrics(sample, metrics=settings['metrics'],
                                     save=False, for_masked=False)
        metrics_masked = runner.get_metrics(sample, metrics=settings['metrics'],
                                            save=False, for_masked=True)
        print("metrics:", metrics)
        print("metrics_masked:", metrics_masked)
        for m in settings['metrics']:
            avg_metrics[m].append(metrics[m])
            avg_metrics_mask[m].append(metrics_masked[m])

    for m in settings['metrics']:
        avg_metrics[m] = sum([float(x) for x in avg_metrics[m]])/len(avg_metrics[m])
        avg_metrics_mask[m] = sum([float(x) for x in avg_metrics_mask[m]])/len(avg_metrics_mask[m])
    
    print("avg_metrics:", avg_metrics)
    print("avg_metrics_mask:", avg_metrics_mask)


    test_dict = {
            settings['model']: {
                settings['test_name']: {
                    "avg_metrics" : avg_metrics,
                    "avg_metrics_mask" : avg_metrics_mask
                }   
            }
        }
    if not os.path.exists(f"{path_to_data}/out"):
        os.mkdir(f"{path_to_data}/out")
    path_to_out = f"{path_to_data}/out/avg_metrics.json"
            
    if os.path.exists(path_to_out):
        fp = open(path_to_out, 'r+')
        prev_dict = json.load(fp)
        if prev_dict.get(settings['model']):
            prev_dict[settings['model']].update(test_dict[settings['model']])
        else:
            prev_dict.update(test_dict)
        fp.seek(0)
        json.dump(prev_dict, sort_keys=True, indent=4, fp=fp)
        fp.truncate() 
    else:
        fp = open(path_to_out, 'w')
        json.dump(test_dict, sort_keys=True, indent=4, fp=fp)
        fp.close() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NN vessel segmentation')
    parser.add_argument('-c', '--config', type=str, help='path_to_config')
    args = parser.parse_args()    
    settings = json.load(open(args.config))
    main(settings)
    