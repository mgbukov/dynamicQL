import LZ_sim_anneal as LZ
import time
import numpy as np
import sys
from matplotlib import pyplot as plt
import plotting
import pickle
import utils
import os


p=[1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0]
plotting.protocol(np.linspace(0,3.5,50),p,show=True)

exit()

def b2(n10,w=10):
    x = np.array(list(np.binary_repr(n10, width=w)),dtype=np.float)
    x[x > 0.5] = 4.
    x[x < 0.5] = -4.
    return x

L=6
dt=0.01
custom_prot=LZ.custom_protocol(
        J=1.0, L=L, hz=1.0, hx_init_state=-2.0, hx_target_state=2.0,
        delta_t=dt, hx_i=-4., hx_max=4., action_set_=[-8.,0.,8.],
        option='fast'
)

# Can just concatenate files at the end too 
#s = time.time()
#for i in range(100):
#r=np.random.randint(1000)
best=-1
p=[4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0,
-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0]
print(len(p))
exit()
protocol=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, -4, 4, -4, 4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4]
print(custom_prot.evaluate_protocol_fidelity(protocol))
protocol_2=[4]*15+[-4]*15
print(custom_prot.evaluate_protocol_fidelity(protocol_2))
exit()

for i in range(len(protocol)):
    protocol[i]*=-1
    fid=custom_prot.evaluate_protocol_fidelity(protocol)
    if fid > best:
        best=fid
        best_p=np.copy(protocol)
    print(fid)
    protocol[i]*=-1

print(list(best_p))
print(best)
print("--- done ---")
print(custom_prot.evaluate_protocol_fidelity(best_p))
#print("fid",fid)
#print("t=%.5f"%(time.time()-s))
#print(data[2394])
exit()