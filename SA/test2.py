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

nQuench = ["000","100","200","300","400","500","1000","2000","3000","5000","10000","50000","100000"]
fidelity=[]
fidelity_std=[]
for nQ in nQuench:
    tmp=[]
    file_name = "data/SA_nStep-0100_nQuench-%s_Ti-0p04_as-0_hxIS-m2p00_hxFS-2p00_deltaT-0p0200_hxI-m4p00_RL-1_L-06_J-1p00_hz-1p00_symm-0.pkl"%nQ
    with open(file_name,'rb') as f:
        #print(file_name)
        data=pickle.load(f)
        series = data[1]
        for d in series:
            tmp.append(d[1])
        #fidelity.append(np.mean(tmp))
        #fidelity.append(np.percentile(tmp,90))
        fidelity.append(np.max(tmp))
        fidelity_std.append(np.std(tmp))

#n=np.array(nQuench,dtype=np.int)
#plt.scatter(n,fidelity)
#plt.show()
print(fidelity)
#print(fidelity_std)
#sns.distplot(np.array(fidelity))
#plt.show()

#fidelity_T.append(np.max(fidelity))
