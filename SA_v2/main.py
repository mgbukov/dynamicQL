'''
Created on May 14 , 2017

@author: Alexandre Day

Purpose: (PYTHON3 IMPLEMENTATION)
    Implements different flavors of simulated annealing for 1D spin chain with uniform hz-field and uniform and time varying hx-field
    Can be run by specifying simulation parameters in the file called 'para.dat'

'''

import utils
import numpy as np
import pickle
from Hamiltonian import HAMILTONIAN
from quspin.operators import exp_op
import time,sys,os
from model import MODEL
#from analysis.compute_observable import MB_observables
    
np.set_printoptions(precision=4)

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
    if parameters['task'] ==  'SA':
        print("Simulated annealing")
        run_SA(parameters, model)
    elif parameters['task'] == 'GB':
        print("Gibbs sampling")
    elif parameters['task'] == 'SD':
        print("Stochastic descent")

    exit()
    

###################################################################################
###################################################################################
###################################################################################
###################################################################################


def run_SA(parameters, model:MODEL, save = True):
    
    if parameters['verbose'] == 0:
        blockPrint()

    n_sample = parameters['n_sample']

    outfile = utils.make_file_name(parameters,root='data/')
    n_exist_sample, all_result = utils.read_current_results(outfile)

    if n_exist_sample >= n_sample :
        print("\n\n-----------> Samples already computed in file -- terminating ... <-----------")
        return all_result

    print("\n\n-----------> Starting simulated annealing <-----------")
    for it in range(n_exist_sample, n_sample):

        start_time=time.time()
        best_fid, best_protocol = SA(parameters, model)
        energy = model.compute_energy(protocol = best_protocol)

        n_fid_eval = parameters['n_quench']

        result = [n_fid_eval, best_fid,  energy, best_protocol]

        print("\n----------> RESULT FOR ANNEALING NO %i <-------------"%(it+1))
        print("Number of fidelity eval \t%i"%n_fid_eval)
        print("Best fidelity \t\t\t%.4f"%best_fid)
        print("Best hx_protocol\t\t",list(best_protocol))
        
        all_result.append(result)
        if save is True:
            with open(outfile,'wb') as f:
                pickle.dump([parameters, all_result],f)
                f.close()
            print("Saved iteration --> %i to %s"%(it,outfile))
        print("Iteration run time --> %.2f s" % (time.time()-start_time))
    
    print("\n Thank you and goodbye !")
    enablePrint()
    return all_result    

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
    n_auto_correlate = n_step*10 # should look at auto-correlation time !
    
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
    energy_samples = []

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
