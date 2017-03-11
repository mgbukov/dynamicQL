import time
import numpy as np
import sys
import pickle
import os
import scipy.io
#scipy.io.savemat('/home/myfiles/mydata.mat', mdict={'whatever_data': whatever_data})

def main():

    if len(sys.argv) > 1:
        _, L, n_step, T = sys.argv
        L = int(L); n_step = int(n_step)
        T = float(T)
    else:
        assert False, "error"

    tot_number_interval = 32
    n_elements = 2**n_step // tot_number_interval
    #hx_int = [n_elements*interval_slice, n_elements*(interval_slice+1)]

    all_data=[]
    for interval_slice in range(tot_number_interval):
        param = {'L' : L, 'T': T, 'n_step': n_step, 'slice': interval_slice}
        file_name = make_file_name(param)
        with open(file_name,'rb') as f:
            data=pickle.load(f)
            
        all_data.append(data)

    mat_file=make_file_name_2(param,ext=".mat")
    scipy.io.savemat(mat_file, mdict={'data': np.concatenate(all_data)})


def make_file_name(param, root="",ext=".pkl",prefix=""):
    key_format = {
        'L':'{:0>2}',
        'T':'{:.3f}',
        'slice':'{:0>2}',
        'n_step':'{:0>2}'
    }

    f = [k+"-"+key_format[k].format(param[k]) for k in sorted(key_format)]
    return root+prefix+'ES_'+"_".join(f)+ext

def make_file_name_2(param, root="",ext=".pkl",prefix=""):
    key_format = {
        'L':'{:0>2}',
        'T':'{:.3f}',
        'n_step':'{:0>2}'
    }

    f = [k+"-"+key_format[k].format(param[k]) for k in sorted(key_format)]
    return root+prefix+'ES_'+"_".join(f)+ext

if __name__ == "__main__":
    main()