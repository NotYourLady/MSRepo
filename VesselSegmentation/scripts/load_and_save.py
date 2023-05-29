from pydicom import dcmread
import pylibjpeg
import os
import numpy as np
import nibabel as nib


def load_nii_vol(path_to_vol, dtype):
    vol_file = nib.load(path_to_vol)
    vol = np.array(vol_file.dataobj, dtype=dtype)
    return(vol, vol_file.affine)
    

def load_sample_data(path_to_sample, dtype):
    head_data = load_nii_vol(path_to_sample + "/head.nii.gz", dtype=dtype)
    vessels_data = load_nii_vol(path_to_sample + "/vessels.nii.gz", dtype=dtype)
    brain_data = load_nii_vol(path_to_sample + "/brain.nii.gz", dtype=dtype)
    
    data = {
        "head" : head_data[0],
        "vessels" : vessels_data[0],
        "brain" : brain_data[0],
        "affine" : head_data[1]
    }
    return(data)


def get_dcm_info(path_to_dcm):
    slices_names = os.listdir(path=path_to_dcm)
    datasets_dict = {}
    for name in slices_names:
        dcm_file = dcmread(path_to_dcm + name)
        sd = dcm_file.SeriesDescription
        if (datasets_dict.get(sd) is None):
            datasets_dict.update({
                sd : {
                    "vol_shape" : None,
                    "vox_size" : None,
                    "file_names" : [name,]
                }    
            })
        else:
            datasets_dict[sd]["file_names"].append(name)
    for key in datasets_dict.keys():
        dcm_file = dcmread(path_to_dcm + datasets_dict[key]["file_names"][0])
        rows = dcm_file.Rows
        columns = dcm_file.Columns
        depth = len(datasets_dict[key]["file_names"])
        sizes = (rows, columns, depth)
        datasets_dict[key]["vol_shape"] = sizes
         
        if hasattr(dcm_file, 'SpacingBetweenSlices') and hasattr(dcm_file, "PixelSpacing"):
            voxel_size = (dcm_file.PixelSpacing[0], dcm_file.PixelSpacing[1], dcm_file.SpacingBetweenSlices)
            datasets_dict[key]["vox_size"] = voxel_size
            
    return(datasets_dict)


def get_dcm_vol(path_to_slices, study_dict):
    vol_shape = study_dict["vol_shape"] 
    vox_size = study_dict["vox_size"]        
    
    dcm_file = dcmread(path_to_slices + study_dict["file_names"][0])
    dtype = type(dcm_file.pixel_array[0][0])
    vol = np.zeros(vol_shape, dtype=dtype)
    
    for indx, name in enumerate(study_dict["file_names"]):
        #print(indx, name)
        dcm_file = dcmread(path_to_slices + name)
        
        vol[:, :, indx] = dcm_file.pixel_array
    
    return(vol, vox_size)


def vox_size2affine(vox_size):
    return [[vox_size[0], 0., 0., 0.],
            [0., vox_size[1], 0., 0.],
            [0., 0., vox_size[2], 0.],
            [0., 0., 0., 1.]]


def save_vol_as_nii(numpy_arr, affine, path_to_save):
    empty_header = nib.Nifti1Header()
    Nifti1Image = nib.Nifti1Image(numpy_arr, affine, empty_header)
    nib.save(Nifti1Image, path_to_save)
  