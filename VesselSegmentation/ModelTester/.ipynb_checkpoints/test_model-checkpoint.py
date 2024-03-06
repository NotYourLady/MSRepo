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
    logger = {}
    for test in tqdm(settings['tests']):
        print("#########################################")
        print("#########################################")
        print(f"Run {test}...")
        print("#########################################")
        print("#########################################")
        
        learner = Learner(settings, test=test)  
        learner.fit_and_save()

        runner = Runner(settings, test=test)
        runner.run()
        logger.update({test: 'done'})
        
        # try:
        #     learner = Learner(settings, test=test)  
        #     learner.fit_and_save()
    
        #     runner = Runner(settings, test=test)
        #     runner.run()
        #     logger.update({test: 'done'})
        # except:
        #     logger.update({test: 'crashed'})
        
    fp = open(f'{settings["results_path"]}/{settings["model"]}_logs.json', 'w')
    json.dump(logger, sort_keys=True, indent=4, fp=fp)
    fp.close()
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NN vessel segmentation')
    parser.add_argument('-c', '--config', type=str, help='path_to_config')
    args = parser.parse_args()    
    settings = json.load(open(args.config))
    main(settings)

