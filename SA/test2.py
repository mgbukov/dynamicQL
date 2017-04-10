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

#######################################
#######################################

fid_SA_group = {}
count = 0 
for T in np.arange(0.1,4.01,0.1):
    fidelity=[]
    fidelity_std=[]
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
            #fidelity.append(np.mean(tmp))
            #fidelity.append(np.percentile(tmp,90))
            fidelity.append(np.max(tmp))
            fidelity_std.append(np.std(tmp))
    fid_group[count]=[fidelity,fidelity_std]
    count+=1

#######################################
#######################################

fid_2SF_group = {}
param = {'L':6,'dt':0.01,'n_step':100,'m':0}
count = 0 
for T in np.arange(0.1,4.01,0.1):
    fidelity=[]
    fidelity_std=[]
    param['dt'] = T/100
    file_name='data/'+utils.make_file_name_2SF(param)
    tmp=[]
    with open(file_name,'rb') as f:
        #print(file_name)
        data = pickle.load(f)
        print(data)
        exit()
        #series = data[1]
        for d in series:
            tmp.append(d[1])
        #fidelity.append(np.mean(tmp))
        #fidelity.append(np.percentile(tmp,90))
        fidelity.append(np.max(tmp))
        fidelity_std.append(np.std(tmp))
    fid_2SF_group[count]=[fidelity,fidelity_std]
    count+=1


max_fid=[]
for c in range(40):
    max_fid.append(np.max(fid_group[c][0]))

print(max_fid)
#n=np.array(nQuench,dtype=np.int)
#plt.scatter(nQ_list,fidelity)
plt.scatter(np.arange(0.1,4.01,0.1),max_fid)
plt.show()
#print(fidelity)
#print(fidelity_std)
#sns.distplot(np.array(fidelity))
#plt.show()

#fidelity_T.append(np.max(fidelity))
