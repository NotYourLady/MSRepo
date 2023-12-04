import os
import json
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname('../.'))
sys.path.append(os.path.dirname('../ml/.'))

from LearnerClass import Learner
from RunnerClass import Runner
from ml.SampleClass import Sample
from scripts.utils import get_path


def main(settings):
    #for test in settings['tests']:
    #    print(settings['tests'][test])
    learner = Learner(settings, test='test1')  
    learner.fit_and_save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NN vessel segmentation')
    parser.add_argument('-c', '--config', type=str, help='path_to_config')
    args = parser.parse_args()    
    settings = json.load(open(args.config))
    main(settings)

