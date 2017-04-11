import LZ_sim_anneal as LZ
import time
import numpy as np
import sys
from matplotlib import pyplot as plt
import plotting
import pickle
import utils
import os
import seaborn as sns
from SD_msector import make_file_name as make_file_name_2SF 

params_SA = utils.default_parameters()
params_SA['L'] = 6
params_SA['N_time_step'] = 100
params_SA['symmetrize'] = False

percentile = 90

fid_SA_group = {}
T_count = 0
for T in np.arange(0.1,4.01,0.1):
    nQ_list =[0,100,200,300,400,500,1000,2000,3000,5000,10000,50000,100000]
    for nQ in nQ_list:
        params_SA['N_quench'] = nQ
        params_SA['delta_t'] = T/100
        file_name='data/'+utils.make_file_name(params_SA)
        tmp=[]
        with open(file_name,'rb') as f:
            #print(file_name)
            data = pickle.load(f)
            series = data[1]
            for d in series:
                tmp.append(d[1])
            #fidelity=np.mean(tmp)
            fidelity=np.percentile(tmp,percentile)
            #fidelity=np.max(tmp)
            fidelity_std=np.std(tmp)
            fid_SA_group[(T_count,nQ)]=[fidelity,fidelity_std]
    T_count+=1

#######################################
#######################################

fid_2SF_group = {}
param = {'L':6,'dt':0.01,'n_step':100,'m':0}
T_count = 0 
for T in np.arange(0.1,4.01,0.1):
    param['dt'] = T/100
    file_name='data/'+make_file_name_2SF(param)
    tmp=[]
    with open(file_name,'rb') as f:
        #print(file_name)
        data = pickle.load(f)
        for d in data:
            tmp.append(d[0])
        #fidelity.append(np.mean(tmp))
        #fidelity.append(np.percentile(tmp,90))
        fidelity = np.max(tmp)
        fidelity_std = np.std(tmp)
    fid_2SF_group[T_count]=[fidelity,fidelity_std]
    T_count+=1


fid_nQ={}
fid_t={}
for nQ in nQ_list:
    fid_nQ[nQ]=[]
for t in range(40):
    fid_t[t]=[]

for t in range(40):
    for nQ in nQ_list:
        f_best = fid_2SF_group[t][0]
        f_SA = fid_SA_group[(t,nQ)][0]
        fid_nQ[nQ].append((f_best-f_SA)/f_best)
        fid_t[t].append((f_best-f_SA)/f_best)

palette = np.array(sns.color_palette('hls',40))
for t in range(4,40,2):
    plt.plot(nQ_list,fid_t[t],label="%.2f"%(0.1*t+0.1),c=palette[t])

#for i,nQ in zip(range(len(nQ_list)),nQ_list):
#    plt.plot(np.arange(0.1,4.01,0.1), fid_nQ[nQ], label=nQ, c=palette[i])

plt.xlabel('$v^{-1}$',fontsize=16)
plt.ylabel('$1-F_{%i}/F_{\mathrm{best}}$'%percentile, fontsize=16)
plt.title('Annealing performance ($L=6$) for different quench velocities \n normalized by the 2 SF fidelities ($F_{\mathrm{best}}$)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.savefig('analysis/SA-F-perc-%i.pdf'%percentile)
#plt.tight_layout()
#plt.clf()

plt.show()


#max_fid=[]
#for c in range(40):
#    max_fid.append(np.max(fid_group[c][0]))

#print(max_fid)
#n=np.array(nQuench,dtype=np.int)
#plt.scatter(nQ_list,fidelity)
#plt.scatter(np.arange(0.1,4.01,0.1),max_fid)
#plt.show()
#print(fidelity)
#print(fidelity_std)
#sns.distplot(np.array(fidelity))
#plt.show()

#fidelity_T.append(np.max(fidelity))
