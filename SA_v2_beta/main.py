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

    # Printing parameters for user
    utils.print_parameters(parameters)

    # Defining Hamiltonian
    H = HAMILTONIAN(**parameters)
    
    # Defines the model, and precomputes evolution matrices given set of states
    model = MODEL(H, parameters)
    
    #root='data/data_ES/'
    root='data/data_SD/'

    
    # Run simulated annealing
    if parameters['task'] ==  'SA':
        print("Simulated annealing")
        run_SA(parameters, model, utils, root)
    elif parameters['task'] == 'GB':
        print("Gibbs sampling") 
        run_GS(parameters, model)
    elif parameters['task'] == 'SD' or parameters['task'] == 'SD2' :
        print("Stochastic descent")
        run_SD(parameters, model, utils, root)
    elif parameters['task'] == 'GRAPE' :
        print("GRAPE")
        run_GRAPE(parameters, model, utils, root)
    elif parameters['task'] == 'ES':
        print("Exact spectrum")
        run_ES(parameters, model, utils, root)
    elif parameters['task'] == 'SASD':
        print("Simulating annealing followed by stochastic descent")
        run_SA(parameters, model, utils, root)


    exit()
    

###################################################################################
###################################################################################
###################################################################################
###################################################################################


def run_SA(parameters, model:MODEL, utils,root, save = True):
    
    if parameters['verbose'] == 0:
        blockPrint()

    outfile = utils.make_file_name(parameters,root=root)
    n_exist_sample, optimal_index, all_result = utils.read_current_results(outfile)
    n_sample = parameters['n_sample']
    if optimal_index is not None:
        best_seen_fid=all_result[optimal_index][1]
    else:
        best_seen_fid=0.0

    if parameters['Ti'] < 0. :
        parameters['Ti'] = compute_initial_Ti(parameters, model, n_sample=1000)
        print("Initial temperature Ti=%.3f" % parameters['Ti'])


    if n_exist_sample >= n_sample :
        print("\n\n-----------> Samples already computed in file -- terminating ... <-----------")

        print("\n\n-------> Best encountered fidelity over all samples is %0.8f <-------"%(best_seen_fid))

        print("\n\n-------> Best encountered hx_protocol over all samples:")
        print(list(all_result[optimal_index][5]))
        print("<-------")

        return  all_result

    print("\n\n-----------> Starting simulated annealing <-----------")
    
    n_iteration_left = n_sample - n_exist_sample  # data should be saved 10 times --> no more (otherwise things are way too slow !)
    n_mod = max([1, n_iteration_left // 10])

    for it in range(n_iteration_left):

        start_time=time.time()
        best_fid, best_protocol, n_fid_eval = SA(parameters, model) # -- --> performing annealing here <-- --
        
        if parameters['task'] == 'SASD':
            print(' -> Stochastic descent ... ')
            model.update_protocol(best_protocol)
            best_fid, best_protocol, n_fid_eval_SD = SD(parameters, model, init = False)
            n_fid_eval += n_fid_eval_SD

        energy, delta_energy, Sent = model.compute_observables(protocol = best_protocol)

        result = [n_fid_eval, best_fid, energy, delta_energy, Sent, best_protocol]

        print("\n----------> RESULT FOR ANNEALING NO %i <-------------"%(it+1))
        print("Number of fidelity eval \t%i"%n_fid_eval)
        print("Best fidelity \t\t\t%.4f"%best_fid)
        print("Best hx_protocol\t\t",list(best_protocol))
        
        all_result.append(result)

        # check if a better fidelity has been seen in previous samples
        if best_fid > best_seen_fid:
            best_seen_fid=best_fid
            optimal_index=len(all_result)-1

        if save and it % n_mod == 0:
            with open(outfile,'wb') as f:
                pickle.dump([parameters, all_result],f)
                f.close()
            print("Saved iteration --> %i to %s"%(it + n_exist_sample,outfile))
        print("Iteration run time --> %.4f s" % (time.time()-start_time))
    

    # print best seen fidelity and protocol over all times
    print("\n\n-------> Best encountered fidelity over all samples is %0.8f <-------"%(best_seen_fid))

    print("\n\n-------> Best encountered hx_protocol over all samples:")
    print(list(all_result[optimal_index][5]))
    print("<-------") 

    print("\n Thank you and goodbye !")
    enablePrint()

    if save :
        with open(outfile,'wb') as f:
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
            model.update_hx(random_time, current_hx)
        
        step += 1
        T = Ti * (1.0-step/n_quench)
    
    return best_fid, best_protocol, n_quench

def run_SD(parameters, model:MODEL, utils, root, save = True):
    
    if parameters['verbose'] == 0:
        blockPrint()

    outfile = utils.make_file_name(parameters,root=root)
    n_exist_sample, optimal_index, all_result = utils.read_current_results(outfile)
    n_sample = parameters['n_sample']
    if optimal_index is not None:
        best_seen_fid=all_result[optimal_index][1]
    else:
        best_seen_fid=0.0

    if n_exist_sample >= n_sample :
        print("\n\n-----------> Samples already computed in file -- terminating ... <-----------")

        print("\n\n-------> Best encountered fidelity over all samples is %0.8f <-------"%(best_seen_fid))

        print("\n\n-------> Best encountered hx_protocol over all samples:")
        print(list(all_result[optimal_index][5]))
        print("<-------")

        return  all_result

    print("\n\n-----------> Starting stochastic descent <-----------")
    
    n_iteration_left = n_sample - n_exist_sample  # data should be saved 10 times --> no more (otherwise things are way too slow !)
    n_mod = max([1,n_iteration_left // 10])
    
    for it in range(n_iteration_left):
       
        start_time=time.time()

        if parameters['task'] == 'SD':
            best_fid, best_protocol, n_fid_eval = SD(parameters, model, init=True) # -- --> performing stochastic descent here <-- -- 
        elif parameters['task'] == 'SD2':
            best_fid, best_protocol, n_fid_eval = SD_2SF(parameters, model, init=True) # -- --> performing 2 spin flip stochastic descent here <-- -- 
        else:
            assert False, 'Error in task specification'

        energy, delta_energy, Sent = model.compute_observables(protocol = best_protocol)

        result = [n_fid_eval, best_fid, energy, delta_energy, Sent, best_protocol]

        print("\n----------> RESULT FOR STOCHASTIC DESCENT NO %i <-------------"%(it+1))
        print("Number of fidelity eval \t%i"%n_fid_eval)
        print("Best fidelity \t\t\t%.16f"%best_fid)
        print("Best hx_protocol\t\t",list(best_protocol))
        
        all_result.append(result)

        # check if a better fidelity has been seen in previous samples
        if best_fid > best_seen_fid:
            best_seen_fid=best_fid
            optimal_index=len(all_result)-1

        if save and it % n_mod == 0:
            with open(outfile,'wb') as f:
                pickle.dump([parameters, all_result],f)
                f.close()
            print("Saved iteration --> %i to %s"%(it + n_exist_sample, outfile))
        print("Iteration run time --> %.4f s" % (time.time()-start_time))
    

    # print best seen fidelity and protocol over all times
    print("\n\n-------> Best encountered fidelity over all samples is %0.8f <-------"%(best_seen_fid))

    print("\n\n-------> Best encountered hx_protocol over all samples:")
    print(list(all_result[optimal_index][5]))
    print("<-------") 

    print("\n Thank you and goodbye !")
    enablePrint()

    if save :
        with open(outfile,'wb') as f:
            pickle.dump([parameters, all_result],f)
            f.close()
    return all_result    

def SD(param, model:MODEL, init=False):
    
    n_step = param['n_step']
    n_fid_eval = 0

    if init:
        # Random initialization
        model.update_protocol( np.random.randint(0, model.n_h_field, size=n_step) )
        old_fid = model.compute_fidelity()
        best_protocol = np.copy(model.protocol())
    else:
        # So user can feed in data say from a specific protocol
        old_fid = model.compute_fidelity()
        best_protocol = np.copy(model.protocol())

    while True: # careful with this. For binary actions, this is guaranteed to break

        random_position = np.arange(n_step, dtype=int)
        np.random.shuffle(random_position)

        local_minima = True
        for t in random_position:
            model.update_hx(t, model.random_flip(t))
            new_fid = model.compute_fidelity()
            n_fid_eval +=1

            if new_fid > old_fid : # accept descent
                old_fid = new_fid
                best_protocol = np.copy(model.protocol())
                local_minima = False # will exit for loop before it ends ... local update accepted
                break
            else:
                model.update_hx(t, model.random_flip(t))
        
        if local_minima:
            break

    return old_fid, best_protocol, n_fid_eval

def SD_2SF(param, model:MODEL, init=False):

    if model.n_h_field > 2:
        assert False, 'This works only for bang-bang protocols'
    
    n_step = param['n_step']
    n_fid_eval = 0

    if init:
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

            model.swap(t1, t2)
            new_fid = model.compute_fidelity()
            n_fid_eval += 1

            if new_fid > old_fid : # accept descent
                #print("%.15f"%new_fid,'\t',n_fid_eval)
                old_fid = new_fid
                best_protocol = np.copy(model.protocol())
                local_minima = False # will exit for loop before it ends ... local update accepted
                break
            else:
                model.swap(t1, t2)
        
        if local_minima:
            break

    return old_fid, best_protocol, n_fid_eval


def GRAPE(param, model:MODEL, init=False):
    
    n_step = param['n_step']
    n_fid_eval = 0

    if init:
        # Random initialization
        model.update_protocol( np.random.randint(0, model.n_h_field, size=n_step) )        
        old_fid = model.compute_fidelity()
        best_protocol = np.copy(model.protocol())

    else:
        # So user can feed in data say from a specific protocol
        old_fid = model.compute_fidelity()
        best_protocol = np.copy(model.protocol())

    # compute random initial protocol
    model.update_protocol( np.random.uniform(param['hx_min'],param['hx_max'],size=param['n_step']) )

    alpha=.9 # initial learning rate
    i_=0
    start_time=time.time()
    while i_<1200: # careful with this. For binary actions, this is guaranteed to break

        # compute protocol gradient
        protocol_gradient=model.compute_protocol_gradient()
        new_protocol=model.protocol() + alpha*protocol_gradient 

        # impose boundedness condition
        ind_max=np.where(new_protocol>param['hx_max'])
        new_protocol[ind_max]=param['hx_max']

        ind_min=np.where(new_protocol<param['hx_min'])
        new_protocol[ind_min]=param['hx_min']

        # update protocol
        model.update_protocol(new_protocol)

        #print(model.protocol())
        #print(model.compute_fidelity(protocol=model.protocol(),discrete=False))
        #print(i_)

        i_+=1
        #alpha/=np.sqrt(i_)

        '''
        random_position = np.arange(n_step, dtype=int)
        np.random.shuffle(random_position)

        local_minima = True
        for t in random_position:
            model.update_hx(t, model.random_flip(t))
            new_fid = model.compute_fidelity()
            n_fid_eval +=1

            if new_fid > old_fid : # accept descent
                old_fid = new_fid
                best_protocol = np.copy(model.protocol())
                local_minima = False # will exit for loop before it ends ... local update accepted
                break
            else:
                model.update_hx(t, model.random_flip(t))
        
        if local_minima:
            break

        '''
    print(time.time()-start_time)
    return old_fid, best_protocol, n_fid_eval


def run_GRAPE(parameters, model:MODEL, utils, root, save = True):
    if parameters['verbose'] == 0:
        blockPrint()

    outfile = utils.make_file_name(parameters,root=root)
    n_exist_sample, optimal_index, all_result = utils.read_current_results(outfile)
    n_sample = parameters['n_sample']
    if optimal_index is not None:
        best_seen_fid=all_result[optimal_index][1]
    else:
        best_seen_fid=0.0

    if n_exist_sample >= n_sample :
        print("\n\n-----------> Samples already computed in file -- terminating ... <-----------")

        print("\n\n-------> Best encountered fidelity over all samples is %0.8f <-------"%(best_seen_fid))

        print("\n\n-------> Best encountered hx_protocol over all samples:")
        print(list(all_result[optimal_index][5]))
        print("<-------")

        return  all_result

    print("\n\n-----------> Starting stochastic descent <-----------")
    
    n_iteration_left = n_sample - n_exist_sample  # data should be saved 10 times --> no more (otherwise things are way too slow !)
    n_mod = max([1,n_iteration_left // 10])
    
    for it in range(n_iteration_left):
       
        start_time=time.time()

        if parameters['task'] == 'GRAPE':
            best_fid, best_protocol, n_fid_eval = GRAPE(parameters, model, init=True) # -- --> performing stochastic descent here <-- -- 
        else:
            assert False, 'Error in task specification'

        exit()

        energy, delta_energy, Sent = model.compute_observables(protocol = best_protocol)

        result = [n_fid_eval, best_fid, energy, delta_energy, Sent, best_protocol]

        print("\n----------> RESULT FOR STOCHASTIC DESCENT NO %i <-------------"%(it+1))
        print("Number of fidelity eval \t%i"%n_fid_eval)
        print("Best fidelity \t\t\t%.16f"%best_fid)
        print("Best hx_protocol\t\t",list(best_protocol))
        
        all_result.append(result)

        # check if a better fidelity has been seen in previous samples
        if best_fid > best_seen_fid:
            best_seen_fid=best_fid
            optimal_index=len(all_result)-1

        if save and it % n_mod == 0:
            with open(outfile,'wb') as f:
                pickle.dump([parameters, all_result],f)
                f.close()
            print("Saved iteration --> %i to %s"%(it + n_exist_sample, outfile))
        print("Iteration run time --> %.4f s" % (time.time()-start_time))
    

    # print best seen fidelity and protocol over all times
    print("\n\n-------> Best encountered fidelity over all samples is %0.8f <-------"%(best_seen_fid))

    print("\n\n-------> Best encountered hx_protocol over all samples:")
    print(list(all_result[optimal_index][5]))
    print("<-------") 

    print("\n Thank you and goodbye !")
    enablePrint()

    if save :
        with open(outfile,'wb') as f:
            pickle.dump([parameters, all_result],f)
            f.close()
    return all_result    



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

def run_ES(parameters, model:MODEL, utils, root):
    
    n_step = parameters['n_step']
    n_protocol = 2**n_step
    exact_data = np.zeros((n_protocol,2), dtype=np.float64) # 15 digits precision

    b2_array = lambda n10 : np.array(list(np.binary_repr(n10, width=n_step)), dtype=np.int)
    st=time.time()
    # ---> measuring estimated time <---
    model.update_protocol(b2_array(0))
    psi = model.compute_evolved_state()
    model.compute_fidelity(psi_evolve=psi)
    model.compute_energy(psi_evolve=psi)
    print("Est. run time : \t %.3f s"%(0.5*n_protocol*(time.time()-st)))
    # ---> Starting real calculation <---

    st=time.time()
    for p in range(n_protocol):
        model.update_protocol(b2_array(p))
        psi = model.compute_evolved_state()
        exact_data[p] = (model.compute_fidelity(psi_evolve=psi), model.compute_energy(psi_evolve=psi))
    
    outfile = utils.make_file_name(parameters,root=root)
    with open(outfile,'wb') as f:
        pickle.dump(exact_data, f, protocol=4)
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
