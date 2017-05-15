
'''
Created on May 5 , 2017

@author: Alexandre Day
'''

import utils
import numpy as np
import pickle
from Hamiltonian import HAMILTONIAN
from quspin.operators import exp_op
import time
from model import MODEL
#from analysis.compute_observable import MB_observables
    
np.set_printoptions(precision=4)

def b2(n10,w=10):
    x = np.array(list(np.binary_repr(n10, width=w)),dtype=np.float)
    x[x > 0.5] = 4.
    x[x < 0.5] = -4.
    return x

def b2_array(n10,w=10):
    return np.array(list(np.binary_repr(n10, width=w)),dtype=np.int)

def main():

    # Reading parameters from para.dat file
    parameters = utils.read_parameter_file()
    
    # Printing parameters for user
    utils.print_parameters(parameters)

    # Defining Hamiltonian
    H = HAMILTONIAN(**parameters)

    # Defines the model, and precomputes evolution matrices given set of states
    model = MODEL(H, parameters)
    
    L = 6
    T = 0.1
    n_step = 28 
    param = {'L' : L, 'T': T, 'n_step': n_step}
    file_name = make_file_name(param, root= "/projectnb/fheating/SGD/ES/dynamicQL/SA/ES/data/")

    with open(file_name, 'rb') as f:
	    fidelities=pickle.load(f)

    nfid=fidelities.shape[0]
    fid_and_energy=np.empty((nfid,2),dtype=np.float)

    for i,f in zip(range(nfid),fidelity):
        if i%10000 == 0: print(i)
        model.update_protocol(b2_array(i, w = 28))
        psi = model.compute_evolved_state()
        fid_and_energy[i][0]=model.compute_fidelity(psi_evolve = psi)
        fid_and_energy[i][1]=model.compute_energy(psi_evolve = psi)
        print(fid_and_energy[0],'\t',f)
        break

    with open("ES_L-06_T-0.500_n_step-28-test.pkl", ‘wb’) as f:
	    fidelities=pickle.dump(fid_and_energy,f, protocol=4)

def make_file_name_2(param, root="",ext=".pkl",prefix=""):
    key_format = {
        'L':'{:0>2}',
        'T':'{:.3f}',
        'n_step':'{:0>2}'
    }

    f = [k+"-"+key_format[k].format(param[k]) for k in sorted(key_format)]
    return root+prefix+'ES_'+"_".join(f)+ext