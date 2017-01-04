'''
Created on Jan 2 , 2017

@author: Alexandre Day

'''

import numpy as np
import pickle
from matplotlib import pyplot as plt
import pandas
import sys
sys.path.append("..")
from plots import plotting
import compute_observable

def main():
    
    compute_time_vs_Nstep=[]
    fid_vs_Nstep=[]
    a_prot_dict={}
    h_prot_dict={}
    fid_dict={}
    n_fid_dict={}
    
    even_file=[2,4,6,8,10,12,14,16,18,20,22,24,30,40,50,60]
    all_file=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,25,30,35,40,45,50,55,60]
    for file_n in even_file:
        file="../data/SA_nStep-%i_nQuench-0_Ti-0p04_as-0_hxIS-m1p00_hxFS-1p00_deltaT-0p05_hxI-m4p00_RL-1_L-1_J-1p24_hz-1p00.pkl"%file_n
        with open(file,'rb') as f:
            [param_SA,result_all]=pickle.load(f)
            n_fid,fid,a_prot,h_prot=split_data(result_all,verbose=False)

            compute_time_vs_Nstep.append([np.mean(n_fid),np.std(n_fid)])
            fid_vs_Nstep.append([np.mean(fid),np.std(fid)])
            
            fid_dict[file_n]=fid
            n_fid_dict[file_n]=fid
            a_prot_dict[file_n]=a_prot
            h_prot_dict[file_n]=h_prot
        
    compute_time_vs_Nstep=np.array(compute_time_vs_Nstep)
    fid_vs_Nstep=np.array(fid_vs_Nstep)
    
    #===========================================================================
    # all_embeddings=[]
    # for file_n in even_file:
    #     print("t-sne for n_time_step=",file_n)
    #     embed=compute_observable.run_tsne(h_prot_dict[file_n])
    #     all_embeddings.append(embed)
    # with open("SGD_tsne_embeddings.pkl","wb") as f:
    #     pickle.dump(all_embeddings,f);
    # print("Done")
    # exit()
    #===========================================================================
    with open("SGD_tsne_embeddings.pkl","rb") as f:
        all_embeddings=pickle.load(f);f.close()
    
    
    for i,file_n in zip(range(len(even_file)),even_file):
        print(file_n)
        xy=all_embeddings[i]
        z=fid_dict[file_n]
        file_name="SGD_tsne_t-%i.pdf"%file_n
        plotting.visne_2D(xy,z,zlabel="Fidelity",out_file=file_name,label="T=%i"%file_n)    
        
         
#===========================================================================
    # EA_OP=[]
    # for file_n in even_file:
    #     print(file_n)
    #     EA_OP.append(compute_observable.Ed_Ad_OP(h_prot_dict[file_n]))
    # 
    # title="Edward-Anderson order parameter ($n=400$) vs. evolution time for SGD\n with the bang-bang8 protocol ($L=1$)"        
    # plotting.observable(np.array(EA_OP),
    #                     np.sort(np.array(list(fid_dict.keys())))*0.05,
    #                     title=title,out_file="SGD_EAOPvsT.pdf",show=True,ylabel="$q_{EA}$"
    #                     )
    #===========================================================================
    
    #===========================================================================
    # title="Std. of fidelity ($n=400$) vs. evolution time for SGD\n with the bang-bang8 protocol ($L=1$)" 
    # plotting.observable(fid_vs_Nstep[:,1],np.sort(np.array(list(fid_dict.keys())))*0.05,title=title,out_file="SGD_stdFvsT.pdf",show=True,ylabel="$\sigma_F$")
    # 
    #===========================================================================
    #===========================================================================
    # 
    # title="Fidelity ($n=400$) vs. evolution time for SGD\n with the bang-bang8 protocol ($L=1$)" 
    # plotting.observable(fid_vs_Nstep[:,0],np.sort(np.array(list(fid_dict.keys())))*0.05,title=title,out_file="SGD_FvsT.pdf",show=True,ylabel="Fidelity")
    # 
    #===========================================================================
    
      
    #===========================================================================
    # title="Computation time ($n=400$) vs. evolution time for SGD\n with the bang-bang8 protocol ($L=1$)" 
    # plotting.observable(compute_time_vs_Nstep[:,0],np.sort(np.array(list(fid_dict.keys())))*0.05,title=title,
    #                     out_file="SGD_compTvsT.pdf",show=True,ylabel="Computation time",xlabel="Evolution Time")
    #===========================================================================
     
    #===========================================================================
    # for n_time_step in [3,7,9,11,13]:
    #     print(n_time_step)
    #     file_name="SGD_t-%i_hxVst.pdf"%n_time_step
    #     plotting.plot_protocol(h_prot_dict[n_time_step][:1],np.arange(0,n_time_step,1)*0.05,title=None,out_file=file_name,labels=['$F$=%.3f'%fid_dict[n_time_step][0]])
    #     
    #===========================================================================
    
    
    
    
    
    #===========================================================================
    # 
    # compute_time_vs_Nstep=np.array(compute_time_vs_Nstep)
    # fid_vs_Nstep=np.array(fid_vs_Nstep)
    # plt.plot(np.arange(5,65,5)*0.05,fid_vs_Nstep[:,1],".-")
    # plt.show()
    # exit()
    # 
    #===========================================================================
    #===========================================================================
    # plt.plot(np.arange(5,65,5)*0.05,fid_vs_Nstep[:,0],".-")
    # plt.show()
    # plt.plot(np.arange(5,65,5)*0.05,compute_time_vs_Nstep[:,0],".-")
    # plt.show()
    # 
    #===========================================================================
#===============================================================================
#     exit()
# 
#     #print(param_SA)
#     print(h_prot)
#     print(fid)
#     
#     #n_fid,fid,a_prot,h_prot=split_data(result_all)
#     exit()
#     plt.hist(fid,bins=np.arange(0.78,0.85,0.005))
#     plt.show()
#===============================================================================
    #print(fid)
    
    
def split_data(result_all,verbose=True):
    N_time_step=len(result_all[0][2])
    N_sample=len(result_all)
    if verbose:
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

