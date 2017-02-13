'''
Created on Jan 22 , 2017

@author: Alexandre Day

Purpose: (PYTHON3 IMPLEMENTATION)
    Computes density of state
'''

import LZ_sim_anneal as LZ
import time
import numpy as np
import pickle


def main():
    
    hx_i=-4.0
    N_time_step=250
    hx_max=4.0
    action_set=[-8.,0,8.0]
    fast_eval=LZ.custom_protocol(delta_t=0.01,option='fast')
    
    random_protocols=unique_protocol(np.array([random_trajectory(hx_i,N_time_step,action_set,hx_max) for _ in range(5000)]))
    start=time.time()
    fidelities=[fast_eval.evaluate_protocol_fidelity(r) for r in random_protocols]
    with open('data/DOS_fid-5000_dt-0p01_Nstep-250.pkl','wb') as f:
        pickle.dump(fidelities,f)
    
    
#============================================================================
# start=time.time()
# print(fast_eval.evaluate_protocol_fidelity(hx_tmp))
# print("Fast ran in %.6f"%(time.time()-start))
#===============================================================================

def unique_protocol(array_of_protocols):
    a=array_of_protocols
    b=np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    return np.unique(b).view(a.dtype).reshape(-1, a.shape[1])

def random_trajectory(hx_i,N_time_step,action_set,hx_max):
    action_protocol=[]
    current_h=hx_i
    for _ in range(N_time_step):    
        while True:
            action_choice=np.random.choice(action_set)
            current_h+=action_choice
            if abs(current_h) < hx_max+0.0001:
                action_protocol.append(action_choice)
                break
            else:
                current_h-=action_choice
    
    return hx_i+np.cumsum(action_protocol)

if __name__ == "__main__":
    main()
    
