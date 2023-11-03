import os
import torchio as tio
from scripts.utils import get_path
from scripts.load_and_save import save_vol_as_nii

class Sample:
    def __init__(self, path_to_sample_dir):
        self.path_to_sample_dir = path_to_sample_dir
        self.sample_name = os.path.basename(path_to_sample_dir)

            
    def get_subject(self, keys=['head', 'vessels', 'brain']):
        sample_dict = {"sample_name" : self.sample_name}
        p = self.path_to_sample_dir
        if ("head" in keys) and get_path(p, "head"):
            sample_dict.update({'head': tio.ScalarImage(get_path(p, "head"))})
        if ("vessels" in keys) and get_path(p, "vessels"):
            sample_dict.update({'vessels': tio.LabelMap(get_path(p, "vessels"))})
        if ("brain" in keys) and get_path(p, "brain"):
            sample_dict.update({'brain': tio.LabelMap(get_path(p, "brain"))})
        
        return(tio.Subject(sample_dict))

        
    def make_masked(self, keys_to_mask, mask_by='brain'):
        p = self.path_to_sample_dir
        masked_path = p + '/masked'
        mask = tio.LabelMap(get_path(p, mask_by)).data
        
        if not os.path.exists(masked_path):
            os.mkdir(masked_path)
        
        for key in keys_to_mask:
            file_path = get_path(p, key)
            file_name = os.path.basename(file_path)
            img_to_save = tio.ScalarImage(file_path)
            arr_to_save = img_to_save.data * mask
            affine = img_to_save.affine
            path_to_save = masked_path +'/masked_' + file_name
            save_vol_as_nii(arr_to_save, affine, path_to_save)    
            
        return(masked_path)