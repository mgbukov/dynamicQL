'''
Created on Jan 2 , 2017

@author: Alexandre Day

Purpose:
    This module is just for testing protocols

'''

import LZ_sim_anneal as LZ
import time
import numpy as np
import sys
from matplotlib import pyplot as plt

#L=2
#hx_tmp=[-4,4]*80
#np.random.seed(0)
#hx_tmp=[4,4,4,4,4,-4,-4,-4,-4]
#hx_tmp2=[4,4,4,4,-4,-4,-4,-4,-4]
#hx_tmp=[[4,-4][np.random.randint(2)] for i in range(160)]
#print(len(hx_tmp))
#plt.step(np.linspace(0,1.6,160),hx_tmp)
#plt.show()


L=4


custom_prot=LZ.custom_protocol(
    J=-1.0,
    L=L, hz=1.0, hx_init_state=-2.0, hx_target_state=2.0,
    delta_t=0.01, hx_i=-4., hx_max=4., action_set_=[-8.,0.,8.],
    option='fast'
)

mag_list=[0,2,4,6,8,10,12,14,16,18,20,22]
best_fid=[]
for magnetization in mag_list:
    print(magnetization)
    fid=[]
    N_step=200
    n_up = int((magnetization + N_step) / 2)
    hx_tmp=np.array([4]*n_up + [-4]*(N_step-n_up))
    for i in range(4000):
        #hx_tmp=[[-4,4][np.random.randint(2)] for i in range(N_step)]
        np.random.shuffle(hx_tmp)
        fid.append(custom_prot.evaluate_protocol_fidelity(hx_tmp))
        #np.random.shuffle(hx_tmp)
    best_fid.append(np.mean(fid))

print(best_fid)
plt.plot(mag_list,best_fid)

#plt.hist(fid, bins=20)
plt.show()

exit()
    
#for i in range(100000):
#    if i%100 ==0: 
#        print(i)
#    hx_tmp=[[-4,4][np.random.randint(2)] for i in range(N_step)]
#    fid.append(custom_prot.evaluate_protocol_fidelity(hx_tmp))

fid=np.array(fid)
print("Best fid is :", np.max(fid))
plt.hist(fid, bins=100)
plt.show()
exit()
start=time.time()
print(standard_eval.evaluate_protocol_fidelity(hx_tmp))
print("Standard run in %.6f"%(time.time()-start))

'''start=time.time()
print(fast_eval.evaluate_protocol_fidelity(hx_tmp))
print("Fast ran in %.6f"%(time.time()-start))
'''