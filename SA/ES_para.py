'''
Created on Mar 8 , 2017

@author: Alexandre Day

Purpose:

    This module is for perforing exhaustive search of 
    fidelities with parallel threads

'''

import LZ_sim_anneal as LZ
import time
import numpy as np
import sys
from matplotlib import pyplot as plt
import plotting
import pickle

def main():

    if len(sys.argv) > 1:
        _, L, n_step, interval_slice, T = sys.argv
        L = int(L); n_step = int(n_step); interval_slice = int(interval_slice)
        T = float(T)
    else:
        L = 6
        n_step = 30
        interval_slice = 0
        T = 2.0

    tot_number_interval = 32
    assert interval_slice < 32, "Wrong !"
    n_elements = 2**n_step // tot_number_interval
    hx_int = [n_elements*interval_slice, n_elements*(interval_slice+1)]

    param = {'L' : L, 'T': T, 'n_step': n_step, 'slice': interval_slice}
    file_name = make_file_name(param, root= "ES/data/")
    print(file_name)

    if L == 1:
        div = 2.
    else:
        div = 1.

    dt = T / n_step
    custom_prot=LZ.custom_protocol(
        J=1.0, hz=1.0, hx_init_state=-2.0/div, hx_target_state=2.0/div,
        L=L, delta_t=dt, 
        hx_i=-4., hx_max=4., action_set_=[-8.,0.,8.], option='fast')

    count = 0
    fid_array = np.ones(n_elements, dtype = np.float)
    for i in range(hx_int[0], hx_int[1]): # long loop !
        fid_array[count] = custom_prot.evaluate_protocol_fidelity( b2(i,n_step) )
        count += 1
    
    best_hx = np.argmax(fid_array)
    best_fid = np.max(fid_array)

    with open(file_name,"wb") as f:   
        pickle.dump(fid_array, f)
        f.close()

    file_name = make_file_name(param, root= "ES/data/",prefix="best_")
    with open(file_name,"wb") as f:
        pickle.dump([best_hx,best_fid],f)
        

def b2(n10,w=10):
    x = np.array(list(np.binary_repr(n10, width=w)),dtype=np.float)
    x[x > 0.5] = 4.
    x[x < 0.5] = -4.
    return x

def make_file_name(param, root="",ext=".pkl",prefix=""):
    key_format = {
        'L':'{:0>2}',
        'T':'{:.3f}',
        'slice':'{:0>2}',
        'n_step':'{:0>2}'
    }

    f = [k+"-"+key_format[k].format(param[k]) for k in sorted(key_format)]
    return root+prefix+'ES_'+"_".join(f)+ext


if __name__ == "__main__":
    main()