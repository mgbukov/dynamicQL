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

# Import pkl file
#file=open('data/allresultsL1.pkl','rb')
#data=pickle.load(file,encoding='latin1') # encoding must be specified if the file was saved with pickle python 2.

#Color palette
palette_ALEX = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']


def main():
    '''
    Example run
    ''' 
    
    # Import pkl file
    t=50
    action_set=1
    L=1
    
    fid_vs_t=[]
    for tt in [5,10,15,20,25,30,35,40,45,50]:
        file=open('data/a1_t%d.pkl'%tt,'rb')
        data=pickle.load(file,encoding='latin1')
        print(np.mean(np.array([d[0] for d in data])))
        fid_vs_t.append([tt*0.05,np.mean(np.array([d[0] for d in data]))])
    
    title="Fidelity vs quench time for $L=1$ with $a\in\\{0,\pm2\\}$, $h_z=1.0$"
    out_file="plots/FidvsT_L=1_a=1.pdf"
    plot_fidelity(np.array(fid_vs_t),title=title,out_file=out_file)

    #===========================================================================
    # file=open('data/a1_t%d.pkl','rb')%(t)
    # data=pickle.load(file,encoding='latin1') # encoding must be specified if the file was saved with pickle python 2.
    # fid=[d[0] for d in data[:10]]
    # 
    # print(np.array([d[0] for d in data]))
    # protocol=[d[2] for d in data[:10]]
    # plot_protocols(protocol,
    #                fid,
    #                title='$t=%.2f,h_z=1.0,action set = %d, L=%d,a\in\\{0,\pm2\\}$'%(t*0.05,action_set,L),
    #                out_file="plots/SA_t=%d_actionset=%d_L=%d.pdf"%(t,action_set,L)
    #                )
    #===========================================================================
def plot_fidelity(fid_vs_time,title,out_file=None):
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family':'serif'})
    plt.plot(fid_vs_time[:,0],fid_vs_time[:,1],'-o',clip_on=False)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Fidelity')
    
    if out_file is not None:
        plt.savefig(out_file)
    plt.show()



def plot_protocols(y_val_list,z_val_list,x_val_list=None,title=None,out_file=None):
    '''
        Shows the plot for the followed protocols. Fidelity is shown in the legend.
        Make constraints -> hx with [-3,1]
    
    ''' 
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family':'serif'})
    
    if x_val_list is None:
        i=0
        for protocol,fidelity in zip(y_val_list,z_val_list):
            N_time_step=len(protocol)
            x=np.linspace(0,N_time_step*0.05,N_time_step)
            y=list(protocol)
            
            # Change plot to step if u want true protocol 
            points=plt.plot(x,y,label=str(round(fidelity,2)))
            i+=1
            
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('$h_x(t)$')
        plt.legend(loc='upper right', shadow=True)
        if out_file is not None:
            plt.savefig(out_file)
        
    plt.show()


# Run main program !
if __name__ == "__main__":
    main()