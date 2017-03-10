import LZ_sim_anneal as LZ
import time
import numpy as np
import sys
from matplotlib import pyplot as plt
import plotting
import pickle

def b2(n10,w=10):
    x = np.array(list(np.binary_repr(n10, width=w)),dtype=np.float)
    x[x > 0.5] = 4.
    x[x < 0.5] = -4.
    return x

with open("data/ES_L-06_T-1.000_n_step-20_slice-10.pkl", "rb") as f:
    data=pickle.load(f)

L=6
dt=1.0/20
n_step = 20
int_size = 2**20//32 

custom_prot=LZ.custom_protocol(
        J=-1.0,
        L=L, hz=1.0, hx_init_state=-2.0, hx_target_state=2.0,
        delta_t=dt, hx_i=-4., hx_max=4., action_set_=[-8.,0.,8.],
        option='fast'
    )

# Can just concatenate files at the end too !
print(custom_prot.evaluate_protocol_fidelity( b2(int_size*10+2394,n_step) ))
print(data[2394])