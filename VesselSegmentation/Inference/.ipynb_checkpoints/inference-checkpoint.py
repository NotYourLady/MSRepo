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
    
    sample_names = []
    for name in os.listdir(path_to_data):
        if not os.path.isfile(f"{path_to_data}/{name}"):
            sample_names.append(name)
            
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
    
    fp1 = open(f"{path_to_data}/avg_metrics_{settings['model']}.json", 'w')
    json.dump(avg_metrics, sort_keys=True, indent=4, fp=fp1)
    fp1.close()
    
    fp2 = open(f"{path_to_data}/avg_metrics_mask_{settings['model']}.json", 'w')
    json.dump(avg_metrics_mask, sort_keys=True, indent=4, fp=fp2)
    fp2.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NN vessel segmentation')
    parser.add_argument('-c', '--config', type=str, help='path_to_config')
    args = parser.parse_args()    
    settings = json.load(open(args.config))
    main(settings)
    