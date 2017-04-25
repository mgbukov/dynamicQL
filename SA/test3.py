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

#with open("data/ES_L-06_T-1.000_n_step-20_slice-10.pkl", "rb") as f:
#    data=pickle.load(f)

#L=6
#dt=1.0/20
#n_step = 20
#int_size = 2**20//32 

#custom_prot=LZ.custom_protocol(
#        J=-1.0,
#        L=L, hz=1.0, hx_init_state=-2.0, hx_target_state=2.0,
#        delta_t=dt, hx_i=-4., hx_max=4., action_set_=[-8.,0.,8.],
#        option='fast'
#    )

# Can just concatenate files at the end too !
#print(custom_prot.evaluate_protocol_fidelity( b2(int_size*10+2394,n_step) ))
#print(data[2394])

pair_fid_T=[]
for n_quench in [0,10,50,100,500,1000,5000,10000,50000,100000]:
    print(n_quench)
    fidelity_T=[]
    valid_T=[]
    for T in np.arange(0.1,4.01,0.1):
        print(T)
        params_SA=utils.default_parameters()
        params_SA['L']=6
        params_SA['N_time_step'] = 200
        params_SA['delta_t'] = T/200.
        params_SA['N_quench'] = n_quench
        params_SA['symmetrize']=False
        file=utils.make_file_name(params_SA)

        fidelity = []
        file_name = 'data/'+file
        if os.path.exists(file_name): 
            with open(file_name,'rb') as f:
                data=pickle.load(f)
                series = data[1]
                for d in series:
                    fidelity.append(d[1])
            fidelity_T.append(np.max(fidelity))
            valid_T.append(T)
    pair_fid_T.append([valid_T,fidelity_T])

for p in pair_fid_T:
    plt.scatter(p[0],p[1],s=5)
plt.show()