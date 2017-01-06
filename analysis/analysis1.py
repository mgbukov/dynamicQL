'''
Created on Jan 2 , 2017

@author: Alexandre Day

'''

import numpy as np
import pickle
from matplotlib import pyplot as plt
import pandas
import sys
import utilities as ut
import compute_observable
sys.path.append("..")
from plots import plotting
from sklearn.decomposition import PCA

def main():
    
    even_file_AS0=[2,4,6,8,10,12,14,16,18,20,22,24,30,40,45,50,55,60]
    file_name_with_replace="../data/SA_nStep-%i_nQuench-0_Ti-0p04_as-0_hxIS-m1p00_hxFS-1p00_deltaT-0p05_hxI-m4p00_RL-1_L-1_J-1p24_hz-1p00.pkl"
    fid_dict_AS0,_,_,h_prot_dict_AS0,_,fid_vs_Nstep_AS0=ut.gather_data(file_name_with_replace,even_file_AS0)
    
    even_file_AS2=[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,35,40,45,50,55,60]
    file_name_with_replace="../data/SA_nStep-%i_nQuench-0_Ti-0p04_as-2_hxIS-m1p00_hxFS-1p00_deltaT-0p05_hxI-m4p00_RL-1_L-1_J-1p24_hz-1p00.pkl"
    fid_dict_AS2,_,_,h_prot_dict_AS2,_,fid_vs_Nstep_AS2=ut.gather_data(file_name_with_replace,even_file_AS2)
    
    fid_vs_Nstep=[fid_vs_Nstep_AS0[:,0],fid_vs_Nstep_AS2[:,0]]
    std_vs_Nstep=[fid_vs_Nstep_AS0[:,1],fid_vs_Nstep_AS2[:,1]]
    
    #print(fid_vs_Nstep)
    xtimes=[np.sort(np.array(list(fid_dict_AS0.keys())))*0.05,np.sort(np.array(list(fid_dict_AS2.keys())))*0.05]
    
    
    
    
    #===========================================================================
    # for n_time_step in [6,12,18,26,30,40,50,60]:
    #     file_name="SGD_t-%i_hxVst_AS2.pdf"%n_time_step
    #     pos_best=np.argmax(fid_dict_AS2[n_time_step])
    #     best_h_prot=h_prot_dict_AS2[n_time_step][pos_best].reshape(1,-1)
    #     print(best_h_prot)
    #     #np.argmax[fid_dict_AS2[n_time_step]]
    #     #print(h_prot_dict_AS2[n_time_step][:1])
    #     plotting.protocol(best_h_prot,np.arange(0,n_time_step,1)*0.05,title=None,out_file=file_name,labels=['$F$=%.3f'%fid_dict_AS2[n_time_step][pos_best]])
    #      
    # 
    #===========================================================================
    
 #==============================================================================
 #    title="Std. of fidelity ($n=400$) vs. evolution time for SGD\n with the different action protocols ($L=1$)" 
 #    plotting.observable(std_vs_Nstep,xtimes,title=title,
 #                        out_file="SGD_stdvsT_AS0-2.pdf",show=True,
 #                        ylabel="$\sigma_F$",xlabel="$t$",labels=['bang-bang8','continuous'])
 # 
 #    exit()
 #    
 #==============================================================================
    
    
    #===========================================================================
    # 
    # EA_OP_AS0=[]
    # for file_n in even_file_AS0:
    #     print(file_n)
    #     EA_OP_AS0.append(compute_observable.Ed_Ad_OP(h_prot_dict_AS0[file_n]))
    # EA_OP_AS2=[]
    # for file_n in even_file_AS2:
    #     print(file_n)
    #     EA_OP_AS2.append(compute_observable.Ed_Ad_OP(h_prot_dict_AS2[file_n]))
    # 
    # with open("../data/EA_OP_AS0-2.pkl",'wb') as f:
    #     pickle.dump([EA_OP_AS0,EA_OP_AS2],f)
    #===========================================================================
#===============================================================================
#     with open("../data/EA_OP_AS0-2.pkl",'rb') as f:
#         [EA_OP_AS0,EA_OP_AS2]=pickle.load(f)
# 
#     title="Edward-Anderson order parameter ($n=400$) vs. evolution time for SGD\n with the different action protocols ($L=1$)"        
#     plotting.observable([EA_OP_AS0,EA_OP_AS2],
#                         xtimes,
#                         title=title,out_file="SGD_EAOPvsT_AS0-2.pdf",show=True,ylabel="$q_{EA}$",
#                         xlabel="$t$",labels=['bang-bang8','continuous']
#                         )
#===============================================================================

#===============================================================================
#     title="Fidelity ($n=400$) vs. evolution time for SGD\n with the different action protocols ($L=1$)" 
#     plotting.observable(fid_vs_Nstep,xtimes,title=title,
#                         out_file="SGD_FvsT_AS0-2.pdf",show=True,
#                         ylabel="fidelity $F$",xlabel="$t$",labels=['bang-bang8','continuous'])
# 
#     exit()
#===============================================================================
    
    
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
   
    #===========================================================================
    # with open("SGD_tsne_embeddings.pkl","rb") as f:
    #     all_embeddings=pickle.load(f);f.close()
    # 
    # 
    # for i,file_n in zip(range(len(even_file)),even_file):
    #     print(file_n)
    #     xy=all_embeddings[i]
    #     z=fid_dict[file_n]
    #     file_name="SGD_tsne_t-%i.pdf"%file_n
    #     plotting.visne_2D(xy,z,zlabel="Fidelity",out_file=file_name,label="T=%i"%file_n)    
    #     
    #===========================================================================
       

  
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
    # title="Fidelity ($n=400$) vs. evolution time for SGD\n with the continuous protocol ($L=1$)" 
    # plotting.observable(fid_vs_Nstep[:,0],np.sort(np.array(list(fid_dict.keys())))*0.05,title=title,
    #                     out_file="SGD_FvsT.pdf",show=True,
    #                     ylabel="fidelity $F$",xlabel="$t$")
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
    
    
if __name__=='__main__':
    main()

