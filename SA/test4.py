import LZ_sim_anneal as LZ
import time
import numpy as np
import sys
from matplotlib import pyplot as plt
import plotting
import pickle
import utils
import os

def b2(n10,w=10):
    x = np.array(list(np.binary_repr(n10, width=w)),dtype=np.float)
    x[x > 0.5] = 4.
    x[x < 0.5] = -4.
    return x

L=6
dt=0.02
custom_prot=LZ.custom_protocol(
        J=-1.0,
        L=L, hz=1.0, hx_init_state=-2.0, hx_target_state=2.0,
        delta_t=dt, hx_i=-4., hx_max=4., action_set_=[-8.,0.,8.],
        option='fast'
)

# Can just concatenate files at the end too 
s=time.time()
#for i in range(100):
#r=np.random.randint(1000)
custom_prot.evaluate_protocol_fidelity(b2(10,10))
print("t=%.5f"%(time.time()-s))
#print(data[2394])
exit()