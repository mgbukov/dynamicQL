'''
Created on Jan 2 , 2017

@author: Alexandre Day

'''

import numpy as np
import pickle
from matplotlib import pyplot as plt
import pandas

def main():
    
    with open("../data/SA_nstep=20_as=1_dt=0p05.pkl",'rb') as f:
        result_all=pickle.load(f)
    
    n_fid,fid,a_prot,h_prot=split_data(result_all)
    
    plt.hist(fid,bins=np.arange(0.78,0.85,0.005))
    plt.show()
    #print(fid)
    
    
def split_data(result_all):
    N_time_step=len(result_all[0][2])
    N_sample=len(result_all)
    print("--- > N_sample=%i,\t N_time_step=%i"%(N_time_step,N_sample))
    action_protocols=np.empty((N_sample,N_time_step),dtype=np.float32)
    hx_protocols=np.empty((N_sample,N_time_step),dtype=np.float32)
    count_fid_eval=np.empty((N_sample,),dtype=np.int32)
    best_fid=np.empty((N_sample,),dtype=np.float32)
    
    for result,i in zip(result_all,range(N_sample)):
        count_fid_eval[i],best_fid[i],action_protocols[i],hx_protocols[i]=result    
        
    return count_fid_eval,best_fid,action_protocols,hx_protocols
 
if __name__=='__main__':
    main()

