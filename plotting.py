'''
Created on Nov 7, 2016

@author: robertday

Description:
    Some plotting functions; ********** Compatible only with Python 3 **********
'''

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# For latex use !
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2015/bin/x86_64-darwin'
cwd = os.getcwd()

file=open('data/allresultsL1.pkl','rb')
data=pickle.load(file,encoding='latin1')

palette_ALEX = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
#sns.set_palette(palette_ALEX)

#print(data)

def plot_protocols(y_val_list,z_val_list,x_val_list=None,title=None):
    #plt.rcParams.update({'font.size': 50})
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family':'serif'})
    
    
    if x_val_list is None:
        i=0
        for protocol,fidelity in zip(y_val_list,z_val_list):
            N_time_step=len(protocol) 
            print(protocol)
            print(fidelity)
            #c=[fidelity]*N_time_step
            #x=np.array(range(N_time_step))
            #y=np.array(list(protocol))
            points=plt.step(range(N_time_step),list(protocol),c=palette_ALEX[i],label=str(round(fidelity,2)))
            i+=1
        plt.title(title)
        plt.xlabel('Time step')
        plt.ylabel('$h_x(t)$')
        plt.legend(loc='upper right', shadow=True)
        #tmp=plt.colorbar(points)
        #tmp.ax.set_y_label("Fidelity")
    
    plt.show()

fid=[d[0] for d in data]
protocol=[d[2] for d in data]

plot_protocols(protocol,fid,title='$h_z=0.5,L=1,a\in\\{0.0,\pm0.02,\pm0.05,\pm0.08,\pm0.1,\pm0.2,\pm0.4,\pm0.8\\}$')
