import time
import numpy as np
import sys
import pickle
import os

def main():

    if len(sys.argv) > 1:
        _, L, n_step, T = sys.argv
        L = int(L); n_step = int(n_step)
        T = float(T)
    else:
        assert False, "error

    tot_number_interval = 32
    assert interval_slice < 32, "Wrong !"
    n_elements = 2**n_step // tot_number_interval
    hx_int = [n_elements*interval_slice, n_elements*(interval_slice+1)]

    for interval_slice in range(tot_number_interval):
        param = {'L' : L, 'T': T, 'n_step': n_step, 'slice': interval_slice}
        file_name = make_file_name(param)
        with open(file_name,'rb') as f:
            data=pickle.load(f)
        txt_file=make_file_name(param,ext=".txt")
        np.savetxt(txt_file,data)
        os.system(txt_file +" >> zzz.txt")

    print(file_name)






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