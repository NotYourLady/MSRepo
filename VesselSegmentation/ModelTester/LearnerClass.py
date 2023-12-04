import torchio as tio
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from scripts.utils import get_path

from ml.tio_dataset import TioDataset
from ml.get_model import get_model
from ml.ControllerClass import Controller
from ml.metrics import ExponentialLogarithmicLoss, DICE_Metric


class Learner:
    def __init__(self, settings, test):
        self.test = test
        self.settings = settings
        self.learning_settings = settings['learning_settings']
        self.device = self.learning_settings['device']
        self.model_name = settings['model']

        self.dataset = self.init_dataset()
        self.controller = self.init_controller()


    def init_dataset(self): 
        BATCH_SIZE_TRAIN = self.learning_settings['batch_size']
        PATCH_SIZE_TRAIN = self.learning_settings['patch_size_train']
        PATCH_SIZE_TEST = self.learning_settings['patch_size_test']
        OVERLAP_TEST = self.learning_settings['overlap_test']
        
        train_settings  = {
            "patch_shape" : PATCH_SIZE_TRAIN,
            "patches_per_volume" : 64,
            "patches_queue_length" : 1440,
            "batch_size" : BATCH_SIZE_TRAIN,
            "num_workers": 4,
            "sampler": "uniform",
        }
        test_settings = {
            "patch_shape" : PATCH_SIZE_TEST,
            "overlap_shape" : OVERLAP_TEST,
            "batch_size" : 1,
            "num_workers": 4,
        }
        #print(self.settings['tests'][self.test]['train'])
        return TioDataset(self.settings["path_to_data"],
                          train_settings=train_settings,
                          test_settings=test_settings,
                          paths = {
                                 'train': self.settings['tests'][self.test]['train'],
                                 'test': self.settings['tests'][self.test]['val'],
                          })
    
    def init_controller(self):
        model = get_model(self.model_name)
        controller_config = {
            "loss" : ExponentialLogarithmicLoss(gamma_tversky=0.5, gamma_bce=0.5, lamb=0.5,
                                                freq = 0.1, tversky_alfa=0.5),
            "metric" : DICE_Metric(),
            'device' : self.device,
            "model" : model,
            "optimizer_fn" : lambda model: Adam(model.parameters(), lr=0.005),
            "sheduler_fn": None, #lambda optimizer: StepLR(optimizer, step_size=1, gamma=0.9),
            "is2d" : self.learning_settings['is2d'],
            'verbose':True,
            'early_stopping': 3,
        }
        return Controller(controller_config)

    def fit_and_save(self):
        epochs = 50
        self.controller.fit(self.dataset, epochs)
        self.controller.save(f"{self.settings['trained_models_path']}/{self.model_name}_{self.test}")
    
    