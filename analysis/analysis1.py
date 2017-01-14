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
from utilities import make_file_name
sys.path.append("..")
from plots import plotting
import seaborn as sns
from sklearn.decomposition import PCA

def main():
    param={'N_time_step':10,
           'N_quench':0,
           'Ti':0.04,
           'action_set':2,
           'hx_initial_state':-1.0,
           'hx_final_state':1.0,
            'delta_t':0.01,
            'hx_i':-4.0,
            'RL_CONSTRAINT':True,
            'L':1,
            'J':1.24,
            'hz':1.0
            }
 
    import os
    
    
    #===========================================================================
    # pca=PCA(n_components=2)
    # param['N_time_step']=10
    # dc=ut.gather_data(param,'../data/')
    # pca.fit(dc['h_protocol']/4.)
    # X=pca.transform(dc['h_protocol']/4.)
    # 
    # plt.scatter(X[:,0],X[:,1])
    # plt.title('PCA, $t=0.1$, continuous protocol')
    # plt.savefig("PCA_AS2_t-0p1.pdf")
    # plt.show()
    # exit()
    #===========================================================================
  
    #===========================================================================
    # dataBB8=[]
    # param['action_set']=0
    # param['N_time_step']=60
    #  
    # param['delta_t']=0.5/60.
    # dc=ut.gather_data(param,'../data/')
    # pca=PCA(n_components=2)
    # pca.fit(dc['h_protocol']/4.)
    # print(pca.explained_variance_ratio_)
    # exit()
    #  
    # param['delta_t']=3.0/60.
    # dc=ut.gather_data(param,'../data/')
    # X=pca.transform(dc['h_protocol']/4.)
    #  
    # title='PCA$_{50}$, $t=3.0$, continuous protocol, nStep$=60$'
    # out_file="PCA_AS0_t-3p0_nStep-60.pdf"
    # plotting.visne_2D(X[:,0],X[:,1],dc['fid'],zlabel="Fidelity",out_file=out_file,title=title,show=True,xlabel='PCA-1',ylabel='PCA-2')
    # 
    #===========================================================================
    #exit()
    #plt.scatter(X[:,0],X[:,1])
    #plt.title('PCA$_{50}$, $t=1.5$, continuous protocol, nStep$=60$')
    #plt.savefig("PCA_AS0_t-0p8_nStep-60.pdf")
    #plt.show()
    #exit()
    # exit()
    
    #===========================================================================
    # param['N_time_step']=2
    # param['action_set']=0
    # dc=ut.gather_data(param,'../data/')
    # print(dc['h_protocol'])
    # exit()
    # dataBB8=[]
    #===========================================================================
#===============================================================================
#     
#     param['action_set']=0
#     param['N_time_step']=60
#     param['delta_t']=0.5/60
#     
#     dc=ut.gather_data(param,'../data/')
#     
#     protocols=dc['h_protocol']
#     #print(np.shape(dc['h_protocol']))
#     sort_f=np.argsort(dc['fid'])[::-1]
#     
#     print(sort_f[0])
#     
#     #protocols[sort_f[0]]
#     
#     best_prot=protocols[sort_f[0:10]]
#     x=np.array(range(60))*1.0/60
#     #print(best_prot.reshape)
#     #print(x.shape)
#     #print(np.array(range(60))*0.1/60)
#     #print(best_prot)
#     #print(np.shape(best_prot))
#     #print(np.shape(np.arange(0.1,3.05,0.1)*0.05))
# 
#     plotting.protocol(protocols[:2],x,labels=dc['fid'][:2],show=True)
#     
#     exit()    
#     
#     
#===============================================================================
    
    param['N_time_step']=100
    param['action_set']=0
    
    dataBB8=[]
    compTime=[]
    x=[]
    #===========================================================================
    # for t in np.arange(0.1,3.05,0.1):
    #     dt=t/param['N_time_step']
    #     param['delta_t']=dt
    #     is_there,dc=ut.gather_data(param,'../data/')
    #     
    #     if is_there:
    #         eaop=compute_observable.Ed_Ad_OP(dc['h_protocol'],4.0)
    #         print(t,eaop,dc['fid'].shape,'\t',np.mean(dc['n_fid']))
    #         compTime.append(np.mean(dc['n_fid']))
    #         dataBB8.append(eaop)
    #         x.append(t)
    #     else:
    #         print("Data not available for %.3f"%dt)
    # 
    #===========================================================================
    #===========================================================================
    # param['action_set']=0
    # param['delta_t']=0.01
    #===========================================================================
    #===========================================================================
    # for i in range(2,300,4):
    #     param['N_time_step']=i
    #     is_there,dc=ut.gather_data(param,'../data/')
    #     if is_there:
    #         eaop=compute_observable.Ed_Ad_OP(dc['h_protocol'],4.0)
    #         print(i,eaop,dc['fid'].shape,'\t',np.mean(dc['n_fid']))
    #         compTime.append(np.mean(dc['n_fid']))
    #         dataBB8.append(eaop)
    #         x.append(i)
    #     else:
    #         print("Data not available for %i"%i)
    # 
    #===========================================================================
    
    #===========================================================================
    # param['N_time_step']=150
    # is_there,dc=ut.gather_data(param,'../data/')
    # x=np.arange(0,150*0.01,0.01)
    # plotting.protocol(dc['h_protocol'][:3],x,labels=dc['fid'][:3],show=True)
    # exit()
    # #x=np.array(range(2,300,4))*0.01
    #===========================================================================
    param['action_set']=2
    param['delta_t']=0.01
    fid_BB=[]
    h_protocol_BB={}
    n_fid_BB=[]
    x=[]
    
    for i in range(2,300,4):
        param['N_time_step']=i
        data_is_available,dc=ut.gather_data(param,'../data/')
        if data_is_available:
            fid_BB.append(np.mean(dc['fid']))
            h_protocol_BB[i]=dc['h_protocol']
            n_fid_BB.append(np.mean(dc['n_fid']))
            x.append(i*param['delta_t'])
            
    title='Just fooling around'
    plotting.observable(fid_BB,x,title=title,ylabel="$F$",labels=['bang-bang'])
    #plotting.protocol(h_protocol_BB[130][20:25],np.arange(0,130,1)*param['delta_t'])
    
    exit()
    
    #pca.fit()
    #===========================================================================
    # dataCONT=[]
    # for t in range(2,300,4):
    #     print(t)
    #     param['N_time_step']=t
    #     dc=ut.gather_data(param,'../data/')
    #     #print(dc['h_protocol'].shape)
    #     eaop=compute_observable.Ed_Ad_OP(dc['h_protocol'],4.0)
    #     print(eaop)
    #     dataCONT.append(eaop)
    #  
    # file="../data/EAOP_"+ut.make_file_name(param)
    # with open(file,'wb') as f:
    #     pickle.dump(dataCONT,f);f.close();
    # 
    # exit()
    # 
    #===========================================================================
    
    #===========================================================================
    # param['action_set']=0
    # dataBB8=[]
    # for t in range(2,300,4):
    #     print(t)
    #     param['N_time_step']=t
    #     dc=ut.gather_data(param,'../data/')
    #     eaop=compute_observable.Ed_Ad_OP(dc['h_protocol'],4.0)
    #     print(eaop)
    #     #print(dc['h_protocol'].shape)
    #     dataBB8.append(eaop)
    # 
    # file="../data/EAOP_"+ut.make_file_name(param)
    # with open(file,'wb') as f:
    #     pickle.dump(dataBB8,f);f.close();
    # 
    # exit()
    #===========================================================================
    
    #===========================================================================
    # param['N_time_step']=298
    # param['action_set']=0
    # file="../data/EAOP_"+ut.make_file_name(param)
    # with open(file,'rb') as f:
    #     dataBB8=pickle.load(f);f.close();
    # 
    # param['action_set']=2
    # f="../data/EAOP_"+ut.make_file_name(param)
    # with open(f,'rb') as file:
    #     dataCONT=pickle.load(file);
    # 
    # time_axis=np.array(range(2,300,4))*0.01
    # title="Edward-Anderson parameter ($n=400$) vs. evolution time for SGD\n with the different action protocols ($L=1$)" 
    # plotting.observable([dataBB8,dataCONT],[time_axis,time_axis],title=title,
    #                      out_file="SGD_EAOPvsT_AS0-2.pdf",show=True,
    #                      ylabel="$q_{EA}$",xlabel="$t$",labels=['bang-bang8','continuous'])
    #===========================================================================
    
    #===========================================================================
    # param['N_time_step']=250
    # dc=ut.gather_data(param,'../data/')
    # sns.distplot(dc['fid'],kde=False,label='$t=%.3f$'%(param['N_time_step']*0.01))
    # plt.legend(loc='best')
    # plt.savefig('SGD_hist_fid_t2p5.pdf')
    # plt.show()
    # exit()
    #===========================================================================
    
    #===========================================================================
    # title="Fidelity ($n=400$) vs. evolution time for SGD\n with the different action protocols ($L=1$)" 
    # plotting.observable(np.array(data),np.array(range(2,300,4))*0.01,title=title,
    #                      out_file="SGD_FvsT_AS2.pdf",show=True,
    #                      ylabel="$F$",xlabel="$t$",labels=['continuous'])
    # 
    #===========================================================================
    
    
    exit()

    #===========================================================================
    # 
    # even_file_AS0=range(2,300,2)
    # file_name_with_replace="../data/SA_nStep-%i_nQuench-000_Ti-0p04_as-2_hxIS-m1p00_hxFS-1p00_deltaT-0p0100_hxI-m4p00_RL-1_L-1_J-1p24_hz-1p00.pkl
    # results_AS0=ut.gather_data(file_name_with_replace,even_file_AS0)
    # fid_dict_AS0=results_AS0["fid"]
    # h_prot_dict_AS0=results_AS0["h_protocol"]
    # fid_vs_Nstep_AS0=results_AS0["fid_vs_nStep"]
    # 
    # even_file_AS2=[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,35,40,45,50,55,60]
    # file_name_with_replace="../data/SA_nStep-%i_nQuench-0_Ti-0p04_as-2_hxIS-m1p00_hxFS-1p00_deltaT-0p05_hxI-m4p00_RL-1_L-1_J-1p24_hz-1p00.pkl"
    # 
    # results_AS2=ut.gather_data(file_name_with_replace,even_file_AS2)
    # fid_dict_AS2=results_AS2["fid"]
    # h_prot_dict_AS2=results_AS2["h_protocol"]
    # fid_vs_Nstep_AS2=results_AS2["fid_vs_nStep"]
    # 
    # 
    # fid_vs_Nstep=[fid_vs_Nstep_AS0[:,0],fid_vs_Nstep_AS2[:,0]]
    # std_vs_Nstep=[fid_vs_Nstep_AS0[:,1],fid_vs_Nstep_AS2[:,1]]
    # 
    # for e in fid_dict_AS2.values():
    #     print(e.shape)
    # exit()
    # 
    # #print(fid_vs_Nstep)
    # xtimes=[np.sort(np.array(list(fid_dict_AS0.keys())))*0.05,np.sort(np.array(list(fid_dict_AS2.keys())))*0.05]
    # 
    # 
    # 
    #===========================================================================
    
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

