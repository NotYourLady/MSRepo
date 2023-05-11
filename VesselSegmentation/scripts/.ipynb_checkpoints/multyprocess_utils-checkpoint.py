import numpy as np
from multiprocessing import Process, Queue
from time import time, sleep
from niblack3d import Niblack3d


def get_vols_for_processes(test_vol, window_size):
    w, h, d = test_vol.shape
    s11 = (w//2, h//2)
    
    vol11 = test_vol[:s11[0] + window_size[0], :s11[1] + window_size[1], :]
    vol12 = test_vol[s11[0] - window_size[0]:, :s11[1] + window_size[1], :]
    vol21 = test_vol[:s11[0] + window_size[0], s11[1] - window_size[1]:, :]
    vol22 = test_vol[s11[0] - window_size[0]:, s11[1] - window_size[1]:, :]

    return{
        "vol11" : vol11,
        "vol12" : vol12,
        "vol21" : vol21,
        "vol22" : vol22,
        "in_size" : test_vol.shape,
        "window_size" : window_size,
        "s11" : s11
    }


def coonect_volumes(volumes_dict):
    s11 = volumes_dict["s11"]
    window_size = volumes_dict["window_size"]
    
    vol = np.zeros(volumes_dict["in_size"])
    vol[:s11[0], :s11[1], :] = volumes_dict["vol11"][:s11[0], :s11[1], :]
    vol[s11[0]:, :s11[1], :] = volumes_dict["vol12"][window_size[0]:, :s11[1], :]
    vol[:s11[0], s11[1]:, :] = volumes_dict["vol21"][:s11[0], window_size[1]:, :]
    vol[s11[0]:, s11[1]:, :] = volumes_dict["vol22"][window_size[0]:, window_size[1]:, :]
    
    return(vol)


def proc(q, algorithm, one_vol_dict):
    binarized = algorithm.binarize(one_vol_dict["vol"])
    one_vol_dict["vol"] = binarized
    q.put(one_vol_dict)

    
def MultuProcMain(vol, window_size=(3, 3, 3), coef_k=0, coef_a=0):
    vol_dict = get_vols_for_processes(vol, window_size)
    
    Niblack = Niblack3d(window_size, coef_k, coef_a)
    
    q = Queue()
    
    p1 = Process(target = proc, args = (q, Niblack, {"vol" : vol_dict["vol11"],
                                            "vol_name" : "vol11"}))
    p2 = Process(target = proc, args = (q, Niblack, {"vol" : vol_dict["vol12"],
                                            "vol_name" : "vol12"}))
    p3 = Process(target = proc, args = (q, Niblack, {"vol" : vol_dict["vol21"],
                                            "vol_name" : "vol21"}))
    p4 = Process(target = proc, args = (q, Niblack, {"vol" : vol_dict["vol22"],
                                            "vol_name" : "vol22"}))
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    
    out_dict = {"in_size" : vol_dict["in_size"],
                "window_size" : vol_dict["window_size"],
                "s11" : vol_dict["s11"]}
    
    for i in range(4):
        d = q.get()
        out_dict.update({d["vol_name"] : d["vol"]})
    
    binarized = coonect_volumes(out_dict)
    return(binarized)