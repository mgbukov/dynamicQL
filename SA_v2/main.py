'''
Created on Sep 1 , 2016

@author: Alexandre Day

Purpose: (PYTHON3 IMPLEMENTATION)
    Implements simulated annealing for 1D spin chain with uniform hz-field and uniform and time varying hx-field
    Can be run from command line
    
Example of use:
    1. Run with default parameters (specified in file)
        $python LZ_sim_anneal
        
    2. Run with optional parameters: 30 quenches, 20 times steps, action_set, outfile, max number of fidelity evaluations,
    time_scale, number of restarts, verbose, symmetrize_protocol:
        $python LZ_sim_anneal.py 30 20 bang-bang8 out.txt 3000 0.05 100 False True
        
    3. Get some help
        $python LZ_sim_anneal -h
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

    # Run simulated annealing
    run_SA(parameters, model)

    exit()
    
    '''with open("ES_L-06_T-0.500_n_step-28.pkl", ‘rb’) as f:
	    fidelities=pickle.load(f)
    nfid=fidelities.shape[0]

    fid_and_energy=np.empty((nfid,2),dtype=np.float)
    for i,f in zip(range(nfid),fidelity):
        model.update_protocol(b2_array(i), w = 28)
        fid_and_energy[i][0]=fid
        fid_and_energy[i][1]=model.compute_energy()

    with open("ES_L-06_T-0.500_n_step-28-test.pkl", ‘wb’) as f:
	    fidelities=pickle.dump(fid_and_energy,f)
    '''
def run_SA(parameters, model:MODEL):

    print("\n\n-----------> Starting simulated annealing <-----------")

    n_sample = parameters['n_sample']
    n_exist_sample = 0
    all_result = []

    for it in range(n_exist_sample, n_sample):

        start_time=time.time()
        best_fid, best_protocol = SA(parameters, model)
        energy = model.compute_energy(protocol = best_protocol)

        n_fid_eval = parameters['n_quench']

        result = [n_fid_eval, best_fid, best_protocol, energy]

        print("\n----------> RESULT FOR ANNEALING NO %i <-------------"%(it+1))
        print("Number of fidelity eval \t%i"%n_fid_eval)
        print("Best fidelity \t\t\t%.4f"%best_fid)
        print("Best hx_protocol\t\t",list(best_protocol))

        all_result.append(result)

        #with open('data/%s'%outfile_name,'wb') as pkl_file:
            ## Here read first then save, stop if reached quota
        #    pickle.dump([dict_to_save_parameters,all_results],pkl_file);pkl_file.close()
            
        #print("Saved iteration --> %i to %s"%(it,'data/%s'%outfile_name))
        print("Iteration run time --> %.2f s"%(time.time()-start_time))
    
    print("\n Thank you and goodbye !")

def SA(param, model:MODEL):
    
    Ti = param['Ti']
    n_quench = param['n_quench']
    if n_quench == 0:
        return
    n_step = param['n_step']
    
    # initial random protocol
    model.update_protocol( np.random.randint(0, model.n_h_field, size=n_step) )
    old_fid = model.compute_fidelity()
    best_fid = old_fid

    T = Ti
    step = 0
    while T > 1e-12:
        beta = 1./T

        random_time = np.random.randint(0,n_step)
        current_hx = model.protocol_hx(random_time)
        model.update_hx(random_time, model.random_flip(random_time))

        new_fid = model.compute_fidelity()
        
        if new_fid > best_fid:
            best_fid = new_fid 
            best_protocol = np.copy(model.protocol()) # makes an independent copy !

        d_fid = new_fid - old_fid 

        if d_fid > 0. : # accept move
            old_fid = new_fid
        elif np.exp(beta*d_fid) > np.random.uniform() : # accept move
            old_fid = new_fid
        else: # reject move
            model.update_hx(random_time, current_hx)
        
        step += 1
        T = Ti * (1.0-step/n_quench)
    
    return best_fid, best_protocol

def Gibbs_Sampling(param, model:MODEL):

    Ti = param['Ti']
    beta = 1./Ti
    n_step = param['n_step']
    n_equilibrate = 10000
    n_auto_correlate = n_step*10
    
    # initial random protocol
    model.update_protocol( np.random.randint(0, model.n_h_field, size=n_step) )
    old_fid = model.compute_fidelity()
    best_fid = old_fid

    for i in range(n_equilibrate):
        
        random_time = np.random.randint(0,n_step)
        current_hx = model.protocol_hx(random_time)
        model.update_hx(random_time, model.random_flip(random_time))

        new_fid = model.compute_fidelity()

        d_fid = new_fid - old_fid 

        if d_fid > 0. : # accept move
            old_fid = new_fid
        elif np.exp(beta*d_fid) > np.random.uniform() : # accept move
            old_fid = new_fid
        else: # reject move
            model.update_hx(random_time, current_hx)

    samples = []
    fid_samples = []
    energy_samples = []

    for i in range(n_sample):
        
        for j in range(n_auto_correlate):
            random_time = np.random.randint(0,n_step)
            current_hx = model.protocol_hx(random_time)
            model.update_hx(random_time, model.random_flip(random_time))

            new_fid = model.compute_fidelity()

            d_fid = new_fid - old_fid 

            if d_fid > 0. : # accept move
                old_fid = new_fid
            elif np.exp(beta*d_fid) > np.random.uniform() : # accept move
                old_fid = new_fid
            else: # reject move
                model.update_hx(random_time, current_hx)
        
        samples.append(np.copy(model.protocol()))
        fid_samples.append(model.compute_fidelity())
        energy_samples.append(model.compute_energy())
        
    return samples, fid_samples, energy_samples

def SD_1SF(param, model, initial_protocol = None):
    ### --- follow up
    # ...
    return 0

def symmetrize_protocol(hx_protocol):
    Nstep=len(hx_protocol)
    half_N=int(Nstep/2)
    for i in range(half_N):
        hx_protocol[-(i+1)]=-hx_protocol[i]

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__
    
# Run main program !
if __name__ == "__main__":
    main()
