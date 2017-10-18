'''
Created on May 14 , 2017

@author: Alexandre Day

Purpose: (PYTHON3 IMPLEMENTATION)
    Implements different flavors of simulated annealing for 1D spin chain with uniform hz-field and uniform and time varying hx-field
    Can be run by specifying simulation parameters in the file called 'para.dat'

'''

from utils import UTILS
import numpy as np
import pickle
from Hamiltonian import HAMILTONIAN
from quspin.operators import exp_op
import time,sys,os
from itertools import product
from model import MODEL
#from analysis.compute_observable import MB_observables
    
np.set_printoptions(precision=4)

def main():
    # Utility object for reading, writing parameters, etc. 
    utils = UTILS()

    # Reading parameters from para.dat file
    parameters = utils.read_parameter_file()
    
    # Command line specified parameters overide parameter file values
    utils.read_command_line_arg(parameters,sys.argv)
    print(parameters)    
    # Printing parameters for user
    utils.print_parameters(parameters)

    # Defining Hamiltonian
    H = HAMILTONIAN(**parameters)

    # Defines the model, and precomputes evolution matrices given set of states
    model = MODEL(H, parameters)
    
    # Run simulated annealing
    if parameters['task'] ==  'SA':
        print("Simulated annealing")
        run_SA(parameters, model, utils)
    elif parameters['task'] == 'GB':
        print("Gibbs sampling") 
        run_GS(parameters, model)
    elif parameters['task'] == 'SD' or parameters['task'] == 'SD2' or parameters['task'] == 'SD2M0':
        print("Stochastic descent with task = %s"%parameters['task']) # options are [SD, SD2, SD2M0]
        run_SD(parameters, model, utils)
    elif parameters['task'] == 'ES':
        print("Exact spectrum")
        run_ES(parameters, model, utils)
    elif parameters['task'] == 'SASD':
        print("Simulating annealing followed by stochastic descent")
        run_SA(parameters, model, utils)
    else:
        print("Wrong task option used")

    exit()
    

###################################################################################
###################################################################################
###################################################################################
###################################################################################


def run_SA(parameters, model:MODEL, utils, save = True):
    
    if parameters['verbose'] == 0:
        blockPrint()

    outfile = utils.make_file_name(parameters,root=parameters['root'])
    n_exist_sample, all_result = utils.read_current_results(outfile)
    n_sample = parameters['n_sample']

    if parameters['Ti'] < 0. :
        parameters['Ti'] = compute_initial_Ti(parameters, model, n_sample=1000)
        print("Initial temperature Ti=%.3f" % parameters['Ti'])

    if n_exist_sample >= n_sample :
        print("\n\n-----------> Samples already computed in file -- terminating ... <-----------")
        return all_result

    print("\n\n-----------> Starting simulated annealing <-----------")
    
    n_iteration_left = n_sample - n_exist_sample  # data should be saved 10 times --> no more (otherwise things are way too slow !)
    n_mod = max([1, n_iteration_left // 10])

    for it in range(n_iteration_left):

        start_time=time.time()
        best_fid, best_protocol, n_fid_eval = SA(parameters, model) # -- --> performing annealing here <-- --
        
        if parameters['task'] == 'SASD':
            print(' -> Stochastic descent ... ')
            model.update_protocol(best_protocol)
            best_fid, best_protocol, n_fid_eval_SD = SD(parameters, model, init_random = False)
            n_fid_eval += n_fid_eval_SD

        energy = model.compute_energy(protocol = best_protocol)
        
        result = [n_fid_eval, best_fid,  energy, best_protocol]

        print("\n----------> RESULT FOR ANNEALING NO %i <-------------"%(it+1))
        print("Number of fidelity eval \t%i"%n_fid_eval)
        print("Best fidelity \t\t\t%.4f"%best_fid)
        print("Best hx_protocol\t\t",list(best_protocol))
        
        all_result.append(result)
        if save and it % n_mod == 0:
            with open(outfile,'wb') as f:
                pickle.dump([parameters, all_result],f)
                f.close()
            print("Saved iteration --> %i to %s"%(it + n_exist_sample,outfile))
        print("Iteration run time --> %.4f s" % (time.time()-start_time))
    
    print("\n Thank you and goodbye !")
    enablePrint()

    if save :
        with open(outfile,'wb') as f:
            print("Saved results in %s"%outfile)
            pickle.dump([parameters, all_result],f)
            f.close()
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
    best_protocol = np.copy(model.protocol())

    T = Ti
    step = 0
    r=0
    while T > 1e-12:
        beta = 1./T
        
        #  --- ---> single spin flip update <--- ---
        random_time = np.random.randint(0,n_step)
        current_hx = model.protocol_hx(random_time)
        model.update_hx(random_time, model.random_flip(random_time))
        #  --- --- --- --- --- --- --- --- --- ---

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
            r+=1
            model.update_hx(random_time, current_hx)
        
        step += 1
        T = Ti * (1.0-step/n_quench)
    
    print(1-r/n_quench)
    return best_fid, best_protocol, n_quench

def run_SD(parameters, model:MODEL, utils, save = True):
    
    if parameters['verbose'] == 0:
        blockPrint()
 
    outfile = utils.make_file_name(parameters,root=parameters['root'])
    n_exist_sample, all_result = utils.read_current_results(outfile)
    n_sample = parameters['n_sample']

    if n_exist_sample >= n_sample :
        print("\n\n-----------> Samples already computed in file -- terminating ... <-----------")
        return all_result

    print("\n\n-----------> Starting stochastic descent <-----------")
    
    n_iteration_left = n_sample - n_exist_sample  # data should be saved 10 times --> no more (otherwise things are way too slow !)
    n_mod = max([1,n_iteration_left // 10])
    fid_series = [-1]
    
    for it in range(n_iteration_left):
       
        start_time=time.time()

        if parameters['task'] == 'SD':
            best_fid, best_protocol, n_fid_eval, n_visit, fid_series = SD(parameters, model, init_random=True) # -- --> performing stochastic descent here <-- -- 
        elif parameters['task'] == 'SD2':
            best_fid, best_protocol, n_fid_eval, n_visit = SD_2SF(parameters, model, init_random=True) # -- --> performing 2 spin flip stochastic descent here <-- -- 
        elif parameters['task'] == 'SD2M0':
            best_fid, best_protocol, n_fid_eval, n_visit = SD_2SF_M0(parameters, model, init_random=True) # -- --> performing 2 spin flip stochastic descent here <-- -- 
        #elif parameters['task'] == 'SD4':
        #best_fid, best_protocol, n_fid_eval, n_enc_state = SD_4SF(parameters, model, init_random=True) # -- --> performing 2 spin flip stochastic descent here <-- -- 
        else:
            assert False, 'Error in task specification'

        energy = model.compute_energy(protocol = best_protocol)
        
        if parameters['fid_series'] is True:
            result = [n_fid_eval, best_fid, energy, n_visit, best_protocol, fid_series]
        else:
            result = [n_fid_eval, best_fid, energy, n_visit, best_protocol, [-1]]
        
        print("\n----------> RESULT FOR STOCHASTIC DESCENT NO %i <-------------"%(it+1))
        print("Number of fidelity eval \t%i"%n_fid_eval)
        print("Number of states visited \t%i"%n_visit)
        print("Best fidelity \t\t\t%.16f"%best_fid)
        print("Best hx_protocol\t\t",list(best_protocol))
        
        all_result.append(result)

        if save and it % n_mod == 0:
            with open(outfile,'wb') as f:
                pickle.dump([parameters, all_result],f)
                f.close()
            print("Saved iteration --> %i to %s"%(it + n_exist_sample, outfile))
        print("Iteration run time --> %.4f s" % (time.time()-start_time))
    
    print("\n Thank you and goodbye !")
    enablePrint()

    if save : # final saving !
        with open(outfile,'wb') as f:
            pickle.dump([parameters, all_result],f)
            print("Saved results in %s"%outfile)
            f.close()
    return all_result    

def SD(param, model:MODEL, init_random=False):
    """ Single spin flip stochastic descent
    """
    
    n_step = param['n_step']
    n_fid_eval = 1
    n_visit = 1

    if init_random:
        # Random initialization
        model.update_protocol( np.random.randint(0, model.n_h_field, size=n_step) )
        old_fid = model.compute_fidelity()
        best_protocol = np.copy(model.protocol())
    else:
        # So user can feed in data say from a specific protocol
        old_fid = model.compute_fidelity()
        best_protocol = np.copy(model.protocol())

    random_position = np.arange(n_step, dtype=int)
    fid_series = [old_fid]

    while True:

        np.random.shuffle(random_position)
        local_minima_reached = True # trick

        for t in random_position:
            
            model.update_hx(t, model.protocol_hx(t)^1) # assumes binary fields
            new_fid = model.compute_fidelity()
            n_fid_eval +=1
            fid_series.append(old_fid)

            if new_fid > old_fid : # accept descent
                old_fid = new_fid
                n_visit += 1
                local_minima_reached = False
                break
            else:
                model.update_hx(t, model.protocol_hx(t)^1) # assumes binary fields

        if local_minima_reached:
            break
    
    return old_fid, np.copy(model.protocol()), n_fid_eval, n_visit, fid_series

''' def SD_symm(param, model:MODEL, init_random=False):
    """ Stochastic descent in symmetrized sector
    """
    
    n_step = param['n_step']
    assert n_step % 2 ==0, 'Must be an even number of time steps to symmetrize ... ABORTING'
    n_fid_eval = 0
    n_visit = 1

    def symmetrize_protocol(hx_protocol):
    Nstep=len(hx_protocol)
    half_N=int(Nstep/2)
    for i in range(half_N):
        hx_protocol[-(i+1)]=-hx_protocol[i]

    if init_random:
        # Random initialization
        tmp = np.random.randint(0, model.n_h_field, size=n_step)


        model.update_protocol(np.random.randint(0, model.n_h_field, size=n_step) )
        old_fid = model.compute_fidelity()
        best_protocol = np.copy(model.protocol())
    else:
        # So user can feed in data say from a specific protocol
        old_fid = model.compute_fidelity()
        best_protocol = np.copy(model.protocol())

    random_position = np.arange(n_step, dtype=int)

    while True:

        np.random.shuffle(random_position)
        local_minima_reached = True # trick

        for t in random_position:
            
            model.update_hx(t, model.protocol_hx(t)^1) # assumes binary fields
            new_fid = model.compute_fidelity()
            n_fid_eval +=1

            if new_fid > old_fid : # accept descent
                old_fid = new_fid
                n_visit += 1
                local_minima_reached = False
                break
            else:
                model.update_hx(t, model.protocol_hx(t)^1) # assumes binary fields
        
        if local_minima_reached:
            break

    return old_fid, np.copy(model.protocol()), n_fid_eval, n_visit '''


def SD_2SF(param, model:MODEL, init_random=False):
    """ 2SF + 1 SF stochastic descent: all possible 2 spin-flip and
    1 spin-flip moves are considered. Algorithm halts when all moves will decrease fidelity
    """

    if model.n_h_field > 2:
        assert False, 'This works only for bang-bang protocols'
    
    n_step = param['n_step']
    n_fid_eval = 0
    n_visit = 1

    if init_random:
        # Random initialization
        model.update_protocol( np.random.randint(0, model.n_h_field, size=n_step) )
        old_fid = model.compute_fidelity()
        best_protocol = np.copy(model.protocol())
    else:
        # So user can feed in data say from a specific protocol
        old_fid = model.compute_fidelity()
        best_protocol = np.copy(model.protocol())

    x1_ar, x2_ar = np.triu_indices(n_step,1)

    n_2F_step = x1_ar.shape[0] # number of possible 2-flip updates
    order2F = np.arange(0,n_2F_step, dtype=np.int) # ordering sequenc // to be randomly shuffled

    n_1F_step = n_step
    order1F = np.arange(0,n_1F_step, dtype=np.int) # ordering sequenc // to be randomly shuffled

    order1F_vs_2F = 2*np.ones(n_2F_step + n_1F_step, dtype=np.int)
    order1F_vs_2F[:n_1F_step]=1
    

    ############################
    #########################
    while True: # careful with this. For binary actions, this is guaranteed to break

        np.random.shuffle(order1F)
        np.random.shuffle(order2F)
        np.random.shuffle(order1F_vs_2F)
        idx_1F=0
        idx_2F=0

        local_minima_reached = True

        for update_type in order1F_vs_2F:
            
            if update_type == 1:    
                # perform 1 SF update
                t = order1F[idx_1F]
                model.update_hx(t, model.protocol_hx(t)^1) # assumes binary fields
                new_fid = model.compute_fidelity()
                n_fid_eval +=1
                idx_1F+=1

                if new_fid > old_fid : # accept descent
                    #print("%.15f"%new_fid,'\t',n_fid_eval)
                    n_visit+=1
                    old_fid = new_fid
                    local_minima_reached = False # will exit for loop before it ends ... local update accepted
                    break
                else:
                    model.update_hx(t, model.protocol_hx(t)^1)
            else:
                # perform 2 SF update
                o2F=order2F[idx_2F]
                t1,t2 = x1_ar[o2F],x2_ar[o2F]
                model.update_hx(t1, model.protocol_hx(t1)^1) # assumes binary fields
                model.update_hx(t2, model.protocol_hx(t2)^1)
                new_fid = model.compute_fidelity()
                n_fid_eval +=1
                idx_2F+=1

                if new_fid > old_fid : # accept descent
                    print("%.15f"%new_fid,'\t',n_fid_eval)
                    n_visit+=1
                    old_fid = new_fid
                    local_minima_reached = False # will exit for loop before it ends ... local update accepted
                    break
                else:
                    model.update_hx(t1, model.protocol_hx(t1)^1) # assumes binary fields
                    model.update_hx(t2, model.protocol_hx(t2)^1)

        if local_minima_reached:
            break

    return old_fid, np.copy(model.protocol()), n_fid_eval, n_visit

def SD_2SF_M0(param, model:MODEL, init_random=False):
    """Two spin flip stochastic descent in the M=0 sector
    """

    if model.n_h_field > 2:
        assert False, 'This works only for bang-bang protocols'
    
    n_step = param['n_step']
    n_fid_eval = 0
    n_visit = 1
    
    if init_random:
        # Random initialization
        tmp = np.ones(n_step,dtype=int) # m = 0 state ...
        tmp[0:n_step//2] = 0
        np.random.shuffle(tmp) 
        
        model.update_protocol(tmp)
        old_fid = model.compute_fidelity()
        best_protocol = np.copy(model.protocol())

    else:
        # So user can feed in data say from a specific protocol
        old_fid = model.compute_fidelity()
        best_protocol = np.copy(model.protocol())
 
    x1_ar, x2_ar = np.triu_indices(n_step,1) 
    order = np.arange(0, x1_ar.shape[0] , dtype=np.int)

    while True: # careful with this. For binary actions, this is guaranteed to break

        np.random.shuffle(order)
        local_minima = True

        for pos in order:
            t1, t2 = (x1_ar[pos], x2_ar[pos])

            if model.protocol_hx(t1) != model.protocol_hx(t2) :
                model.swap(t1, t2)
                new_fid = model.compute_fidelity()
                n_fid_eval += 1

                if new_fid > old_fid : # accept descent
                    #print("%.15f"%new_fid,'\t',n_fid_eval)
                    old_fid = new_fid
                    n_visit +=1
                    #best_protocol = np.copy(model.protocol())
                    local_minima = False # will exit for loop before it ends ... local update accepted
                    break
                else:
                    model.swap(t1, t2)
            
        if local_minima:
            break

    return old_fid, np.copy(model.protocol()), n_fid_eval, n_visit


def Gibbs_Sampling(param, model:MODEL): 
    # should also measure acceptance rate 

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

def compute_initial_Ti(param, model:MODEL, n_sample = 100, rate = 0.8):
    # OK how is this acceptable ? >>>>>>> not tested at all <<<<<<<<
    # Estimates the high-temperature limit (where the acceptance rate is 99 the average worst case excitations %) 

    n_step = param['n_step']
    dF_mean = []

    for _ in range(n_sample):
        model.update_protocol( np.random.randint(0, model.n_h_field, size=n_step) )
        old_fid = model.compute_fidelity()
        rand_pos = np.random.randint(n_step)
        model.update_hx(rand_pos, model.random_flip(rand_pos))
        dF = model.compute_fidelity()-old_fid
        if dF < 0: 
            dF_mean.append(dF)
    
    return np.mean(dF_mean)/ np.log(rate)

def run_ES(parameters, model:MODEL, utils):
    
    n_step = parameters['n_step']
    n_protocol = 2**n_step
    exact_data = np.zeros((n_protocol,3), dtype=np.float64) # 15 digits precision

    b2_array = lambda n10 : np.array(list(np.binary_repr(n10, width=n_step)), dtype=np.int)
    st=time.time()
    # ---> measuring estimated time <---
    model.update_protocol(b2_array(0))
    psi = model.compute_evolved_state()
    model.compute_fidelity(psi_evolve=psi)
    model.compute_Sent(psi_evolve=psi)
    model.compute_energy(psi_evolve=psi)
    print("Est. run time : \t %.3f s"%(0.5* n_protocol*(time.time()-st)))
    # ---> Starting real calculation <---

    st=time.time()
    for p in range(n_protocol):
        model.update_protocol(b2_array(p))
        psi = model.compute_evolved_state()
        exact_data[p] = (model.compute_fidelity(psi_evolve=psi), model.compute_energy(psi_evolve=psi), model.compute_Sent(psi_evolve=psi))
    
    outfile = utils.make_file_name(parameters, root=parameters['root'])
    with open(outfile,'wb') as f:
        pickle.dump(exact_data, f, protocol=4)

    print("Saved results in %s"%outfile)
    print("Total run time : \t %.3f s"%(time.time()-st))
    print("\n Thank you and goodbye !")
    f.close()


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
