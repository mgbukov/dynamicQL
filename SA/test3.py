import LZ_sim_anneal as LZ
import time
import numpy as np
import sys
from matplotlib import pyplot as plt
import plotting
import pickle
import utils
from utils import default_parameters
from utils import make_file_name
import os

def b2(n10,w=10):
    x = np.array(list(np.binary_repr(n10, width=w)),dtype=np.float)
    x[x > 0.5] = 4.
    x[x < 0.5] = -4.
    return x

par=default_parameters()
par['symmetrize']=False
par['L']=6
    
best_fid=0
best_fid_list=[]
for n in range(10,405,10):
    par['N_time_step'] = n
    file=make_file_name(par)
    with open('data/'+file, 'rb') as f:
        data = pickle.load(f)
    for d in data[1]:
        if d[1] > best_fid:
            best_fid = d[1]
    best_fid_list.append(best_fid)

np.savetxt('out.txt',best_fid_list)
#print(data)   

exit()
h=data[0][1]
dt=0.015
t=np.arange(0,3.0,dt)
plotting.protocol(t,h)
exit()


#with open("data/ES_L-06_T-1.000_n_step-20_slice-10.pkl", "rb") as f:
#    data=pickle.load(f)

#L=6
#dt=1.0/20
#n_step = 20
#int_size = 2**20//32 

#custom_prot=LZ.custom_protocol(
#        J=1.0,
#        L=L, hz=1.0, hx_init_state=-2.0, hx_target_state=2.0,
#        delta_t=dt, hx_i=-4., hx_max=4., action_set_=[-8.,0.,8.],
#        option='fast'
#    )

# Can just concatenate files at the end too !
#print(custom_prot.evaluate_protocol_fidelity( b2(int_size*10+2394,n_step) ))
#print(data[2394])
#SA_L-06_dt-0.0100_m-000_n_step-0030.pkl

dt=0.01
params_SA=utils.default_parameters()
params_SA['L']=6
params_SA['N_time_step'] = 200
params_SA['delta_t'] = dt
params_SA['N_quench'] = 0
params_SA['symmetrize']=True

file="data/"+utils.make_file_name(params_SA)
file="data/"+"SA_L-06_dt-0.0100_m-000_n_step-0100.pkl"
with open(file,'rb') as f:
    data=pickle.load(f)

#print(data)
#plotting(plotting.protocol(np.arange(0,2.0,dt), data[0][1]))
print(data[0][0])
np.savetxt("here.txt",(data[0][1]).astype(int)[None],fmt='%i',delimiter=',')
exit()
samples=data[1]
fidelities=[]
protocols=[]
best_fid = -1
for s in samples:
    fidelities.append(s[1])
    protocols.append(s[3])
    if s[1] > best_fid:
        best_fid = s[1]
        best_protocol = s[3]

#time=np.arange(0,0.3,0.01)
i=0
print(best_fid)
#print(list(best_protocol.astype(int)))
#plotting.protocol(np.arange(0,1.0,dt), best_protocol)
np.savetxt("here.txt",np.transpose(best_protocol.astype(int))[None],fmt='%i',delimiter=',')
#print(np.unique(fidelities))
#print(protocols)
exit()

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