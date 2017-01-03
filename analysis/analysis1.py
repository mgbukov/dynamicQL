'''
Created on Jan 2 , 2017

@author: Alexandre Day

'''

import numpy as np
import pickle
from matplotlib import pyplot as plt
import pandas

def main():
    
    compute_time_vs_Nstep=[]
    fid_vs_Nstep=[]
    for file_n in range(5,65,5):
        file="../data/SA_nStep-%i_nQuench-0_Ti-0p04_as-0_hxIS-m1p00_hxFS-1p00_deltaT-0p05_hxI-m4p00_RL-1_L-1_J-1p24_hz-1p00.pkl"%file_n
        with open(file,'rb') as f:
            [param_SA,result_all]=pickle.load(f)
            n_fid,fid,a_prot,h_prot=split_data(result_all)
            print(param_SA['N_time_step'])
            print(h_prot[:2])
            compute_time_vs_Nstep.append([np.mean(n_fid),np.std(n_fid)])
            fid_vs_Nstep.append([np.mean(fid),np.std(fid)])
            
    exit()               
    compute_time_vs_Nstep=np.array(compute_time_vs_Nstep)
    fid_vs_Nstep=np.array(fid_vs_Nstep)
    plt.plot(np.arange(5,65,5)*0.05,fid_vs_Nstep[:,1],".-")
    plt.show()
    exit()
    
    plt.plot(np.arange(5,65,5)*0.05,fid_vs_Nstep[:,0],".-")
    plt.show()
    plt.plot(np.arange(5,65,5)*0.05,compute_time_vs_Nstep[:,0],".-")
    plt.show()
    
    exit()

    #print(param_SA)
    print(h_prot)
    print(fid)
    
    #n_fid,fid,a_prot,h_prot=split_data(result_all)
    exit()
    plt.hist(fid,bins=np.arange(0.78,0.85,0.005))
    plt.show()
    #print(fid)
    
    
def split_data(result_all):
    N_time_step=len(result_all[0][2])
    N_sample=len(result_all)
    print("--- > N_sample=%i,\t N_time_step=%i"%(N_sample,N_time_step))
    action_protocols=np.empty((N_sample,N_time_step),dtype=np.float32)
    hx_protocols=np.empty((N_sample,N_time_step),dtype=np.float32)
    count_fid_eval=np.empty((N_sample,),dtype=np.int32)
    best_fid=np.empty((N_sample,),dtype=np.float32)
    
    for result,i in zip(result_all,range(N_sample)):
        count_fid_eval[i],best_fid[i],action_protocols[i],hx_protocols[i]=result    
        
    return count_fid_eval,best_fid,action_protocols,hx_protocols
 
if __name__=='__main__':
    main()

