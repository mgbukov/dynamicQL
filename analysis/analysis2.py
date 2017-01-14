'''
Created on Jan 14 , 2017

@author: Alexandre Day

'''

import numpy as np
import pickle
import utilities as ut
import compute_observable
from utilities import make_file_name
import sys
sys.path.append("..")
from plots import plotting

def main():
    
    "These are some default parameters"
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
 
    
    """ 
    Here you chose to be more specific:
        action_set 0 is the bang-bang
        action_set 2 is the continuous    
    """
 
    """ 
    Selecting simulation data by the parameters that were used
    
    """
    param['action_set']=2
    param['delta_t']=0.01
    
    
    mean_fid_BB=[]
    h_protocol_BB={}
    fid_BB={}
    n_fid_BB=[]
    x=[]
    sigma_fid=[]
    EA_OP=[]
    
    """ 
    Here we loop of the time steps ... 50 to 300 by increments of 4.
    For every time step we stop the data in the various containers.
    Every time step has 4 keys:
        fid,n_fid,h_protocol,a_protocol
    """
    for i in range(50,300,4):
        param['N_time_step']=i
        data_is_available,dc=ut.gather_data(param,'../data/')
        if data_is_available:
            mean_fid_BB.append(np.mean(dc['fid']))
            sigma_fid.append(np.std(dc['fid']))
            fid_BB[i]=dc['fid']
            EA_OP.append(compute_observable.Ed_Ad_OP(dc['h_protocol'],4.0))
            h_protocol_BB[i]=dc['h_protocol']
            n_fid_BB.append(np.mean(dc['n_fid']))
            x.append(i*param['delta_t'])
     
    """ Plotting the fidelity and EA order param., then plotting a few protocols """
    """ For more options (like saving, adding labels, etc, just look at plotting.py) """
        
    plotting.observable([mean_fid_BB,EA_OP],[x,x],title="A simple plot",ylabel="$F$",xlabel="$T$",marker="-")
    
    plotting.protocol(h_protocol_BB[130][20:25],np.arange(0,130,1)*param['delta_t'])
    
    exit()
    

if __name__=='__main__':
    main()

