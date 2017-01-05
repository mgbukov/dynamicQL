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
    time_scale, number of restarts, verbose:
        $python LZ_sim_anneal.py 30 20 bang-bang8 out.txt 3000 0.05 100 False
        
    3. Get some help
        $python LZ_sim_anneal -h
'''

import utilities as ut
import sys,os # for running in batch from terminal
ut.check_sys_arg(sys.argv)
ut.check_version()


import numpy as np
import pickle
import Hamiltonian_alex as Hamiltonian
from quspin.operators import exp_op
import time
import math
from scipy.sparse.linalg import expm_multiply as expm

np.set_printoptions(precision=4)
    
def main():
    
    global action_set,hx_discrete,hx_max,FIX_NUMBER_FID_EVAL
    
    continuous=[0.01,0.05,0.1,0.2,0.5,1.,2.,3.,4.,8.]
    action_set_name=["bang-bang8","continuous-pos","continuous"]
    action_set_arrays=[
                      np.array([-8.0,0.,8.]),
                      np.array(continuous,dtype=np.float32),
                      np.array([-c for c in continuous]+[0]+continuous,dtype=np.float32)   
                      ]
    all_action_sets=dict(zip(action_set_name,action_set_arrays))
    
    """ 
    Parameters
        L: system size
        J: Jzz interaction
        hz: longitudinal field
        hx_i: initial tranverse field coupling
        hx_initial_state: initial state transverse field
        hx_final_state: final state transverse field
        
        N_quench: number of quenches (i.e. no. of time temperature is quenched to reach exactly T=0)
        N_time_step: number of time steps
        action_set: array of possible actions
        outfile_name: file where data is being dumped (via pickle) 
        max_fid_eval: maximum number of fidelity evaluations
        delta_t: time scale
        N_restart: number of restart for the annealing
        verbose: If you want the program to print to screen the progress
        
        hx_max : maximum hx field (the annealer can go between -hx_max and hx_max
        FIX_NUMBER_FID_EVAL: decide wether you want to fix the maximum number of fidelity evaluations
        RL_CONSTRAINT: use reinforcement learning constraints or not
    """
    #----------------------------------------
    # DEFAULT PARAMETERS
    L = 1 # system size
    J = 1.0/0.809 # zz interaction
    hz = 1.0 #0.9045/0.809 #1.0 # hz field
    hx_i = -4.0# -1.0 # initial hx coupling
    hx_initial_state= -1.0 # initial state
    hx_final_state = 1.0 #+1.0 # final hx coupling
    act_set_name='bang-bang8'
    
    N_quench=5
    N_time_step=20
    outfile_name='first_test.pkl'
    action_set=all_action_sets['bang-bang8']
    max_fid_eval=3000
    delta_t=0.05
    N_restart=5
    
    hx_max=4
    FIX_NUMBER_FID_EVAL=False # this fixes the number of quenches automatically, supersedes N_quench 
    RL_CONSTRAINT=True 
    verbose=True
    
    #----------------------------------------
    
    print("-------------------- > Parameters < --------------------")
    print("L \t\t\t %i\nJ \t\t\t %.3f\nhz \t\t\t %.3f\nhx(t=0) \t\t %.3f\nhx_max \t\t\t %.3f "%(L,J,hz,hx_i,hx_max))
    print("hx_initial_state \t %.2f\nhx_final_state \t\t %.2f"%(hx_initial_state,hx_final_state))
    print("N_quench \t\t %i\ndelta_t \t\t %.2f\nN_restart \t\t %i"%(N_quench,delta_t,N_restart))
    
    if len(sys.argv)>1:
        """ 
            if len(sys.argv) > 1 : run from command line -- check command line for parameters 
        """        
        N_quench,N_time_step,action_set,outfile_name,max_fid_eval,delta_t,N_restart,verbose,act_set_name=ut.read_command_line_arg(sys.argv,all_action_sets)
        
    print("N_time_step \t\t %i"%N_time_step)
    print("Total_time \t\t %.2f"%(N_time_step*delta_t))
    print("Output file \t\t %s"%('data/'+outfile_name))
    print("max_fid_eval (%s) \t %i"%(str(FIX_NUMBER_FID_EVAL),max_fid_eval))
    print("Action_set \t <- \t %s"%np.round(action_set,3))
    print("# of possible actions \t %i"%len(action_set))
    print("Fixing no of fid eval \t %s"%str(FIX_NUMBER_FID_EVAL))
    print("Using RL constraints \t %s"%str(RL_CONSTRAINT))

    param={'J':J,'hz':hz,'hx':hx_i} # Hamiltonian kwargs 
    hx_discrete=[0]*N_time_step # dynamical part at every time step (initiaze to zero everywhere)
    
    # full system hamiltonian
    H,_ = Hamiltonian.Hamiltonian(L,fct=hx_vs_t,**param)
   # print(H)
   
    # calculate initial and final states
    hx_discrete[0]=hx_initial_state # just a trick to get initial state
    E_i, psi_i = H.eigsh(time=0,k=1,which='SA')
    hx_discrete[0]=hx_final_state # just a trick to get final state
    E_f, psi_target = H.eigsh(time=0,k=1,which='SA')
    hx_discrete[0]=0

    
    sweep_size=N_time_step*len(action_set)
    if FIX_NUMBER_FID_EVAL:
        # Determines the number of quenches needed to anneal w/o exceeding the max number of fidelity evaluations.
        N_quench=(max_fid_eval-5*sweep_size)//sweep_size
        assert N_quench >= 0
        print("N_quench (FIX)\t\t%i"%N_quench)

    
    # simulated annealing kwargs:
    param_SA={'Ti':0.04,'sweep_size':sweep_size,
              'psi_i':psi_i,'H':H,'N_time_step':N_time_step,
              'delta_t':delta_t,'psi_target':psi_target,
              'hx_i':hx_i,'N_quench':N_quench,'RL_CONSTRAINT':RL_CONSTRAINT,
              'verbose':verbose,'hx_initial_state':hx_initial_state,'hx_final_state':hx_final_state,
              'L':L,'J':J,'hz':hz,'action_set':action_set_name.index(act_set_name)
            }
    
    if outfile_name=="auto": outfile_name=ut.make_file_name(param_SA)
    
    to_save_par=['Ti','sweep_size','psi_i','N_time_step',
                'delta_t','psi_target','hx_i','N_quench','RL_CONSTRAINT',
                'hx_initial_state','hx_final_state','L','J','hz','action_set'
                ]
    
    dict_to_save_parameters=dict(zip(to_save_par,[param_SA[p] for p in to_save_par]))
    
    all_results=[]
    
    for it in range(N_restart):
        print("\n\n-----------> Starting new iteration <-----------")
        start_time=time.time()
    
        count_fid_eval,best_fid,best_action_protocol,best_hx_discrete=simulate_anneal(param_SA)
        result=count_fid_eval,best_fid,best_action_protocol,best_hx_discrete
        print("\n----------> RESULT FOR ANNEALING NO %i <-------------"%(it+1))
        print("Number of fidelity eval \t%i"%count_fid_eval)
        print("Best fidelity \t\t\t%.4f"%best_fid)
        print("Best action_protocol\t\t",best_action_protocol)
        print("Best hx_protocol\t\t",best_hx_discrete)
        
        all_results.append(result)
        
        with open('data/%s'%outfile_name,'wb') as pkl_file:
            pickle.dump([dict_to_save_parameters,all_results],pkl_file);pkl_file.close()
            
        print("Saved iteration --> %i to %s"%(it,'data/%s'%outfile_name))
        print("Iteration run time --> %.2f s"%(time.time()-start_time))
    
    
def Fidelity(psi_i,H,N_time_step,delta_t,psi_target):
    """
    Purpose:
        Calculates final fidelity by evolving psi_i over a N_time_step 
    Return: 
        Norm squared between the target state psi_target and the evolved state (according to the full hx_discrete protocol)
        
    """    
    psi_evolve=psi_i.copy()
    for t in range(N_time_step):
        psi_evolve = exp_op(H(time=t),a=-1j*delta_t).dot(psi_evolve)
    
    #print(psi_evolve)
    return abs(np.sum(np.conj(psi_evolve)*psi_target))**2

# Returns the hx field at a given time slice 
def hx_vs_t(time): return hx_discrete[int(time)]

def random_trajectory(hx_i,N_time_step):
    '''
    Purpose:
        Generates a random trajectory
    Return:
        Action protocol,hx_discrete ; action taken at every time slice, field at every time slice
    '''
    action_protocol=[]
    current_h=hx_i
    for _ in range(N_time_step):    
        while True:
            action_choice=np.random.choice(action_set)
            current_h+=action_choice
            if abs(current_h) < hx_max+0.0001:
                action_protocol.append(action_choice)
                break
            else:
                current_h-=action_choice
    
    return action_protocol,hx_i+np.cumsum(action_protocol)

# Check if two floats are equal according to some precision
def are_equal(a,b,prec=0.000001):
    return abs(a-b)<prec

# Check if a is in set by comparing floats (with a given precision)
def is_element_of_set(a,set,prec=0.000001):
    return np.min(abs(set-a)) < prec

def avail_action_RL(time_position,old_action_protocol,old_hx_discrete,hx_i,N_time_step): # simulate SA in the space of actions, not field... need to recompute traj. 
    """
    Purpose:
        Get available actions at a specific time slice given a protocol and with RL constratins 
    Return:
        A list of available actions
    """
    action_subset=[]
    if time_position==N_time_step-1:
        h_pre=old_hx_discrete[time_position-1]
        for a in action_set:
            if abs(a+h_pre) < hx_max+0.00001:
                action_subset.append(a)
    else:
        # General case
        h_next=old_hx_discrete[time_position+1]
        if time_position==0:
            h_pre=hx_i
        else:
            h_pre=old_hx_discrete[time_position-1]
        dh=h_next-h_pre
        for a in action_set:
            if is_element_of_set(dh-a,action_set):
                if abs(a+h_pre) < hx_max+0.00001:
                    action_subset.append(a)
    return action_subset

def avail_action(time_position,old_action_protocol,old_hx_discrete,hx_i,N_time_step):
    """
    Purpose:
        Get available actions at a specific time slice given a protocol without any constraints (except having abs(hx)<abs(hx_max) 
    Return:
        A list of available actions
    """
    action_subset=[]
    if time_position==N_time_step-1:
        h_pre=old_hx_discrete[time_position-1]
        for a in action_set:
            if abs(a+h_pre) < hx_max+0.00001:
                action_subset.append(a)
    else:
        # General case
        if time_position==0:
            h_pre=hx_i
        else:
            h_pre=old_hx_discrete[time_position-1]
        for a in action_set:
            if abs(a+h_pre) < hx_max+0.00001:
                action_subset.append(a)
    return action_subset

def propose_new_trajectory(old_action_protocol,old_hx_discrete,hx_i,N_time_step,RL_constraint=False,rand_pos=None):
    '''
    Purpose:
        Given the old_action_protocol, makes a random change and returns the new action protocol
        
    Return:
        New action protocol,New hx as a function of the time slice
    '''
    new_hx_discrete=np.copy(old_hx_discrete)
    if rand_pos==None:
        rand_pos=np.random.randint(N_time_step)
    current_action=old_action_protocol[rand_pos]
    
    tmp=[]
    count=0
    # w/o RL constraints
    while True:
        if RL_constraint:
            aset=avail_action_RL(rand_pos,old_action_protocol,old_hx_discrete,hx_i,N_time_step)
        else:
            aset=avail_action(rand_pos,old_action_protocol,old_hx_discrete,hx_i,N_time_step)
        for aa in aset:
            if not are_equal(aa,current_action):
                tmp.append(aa)
    
        if len(tmp)==0:
            rand_pos=np.random.randint(N_time_step)
            current_action=old_action_protocol[rand_pos]
        else:
            a=np.random.choice(tmp)
            break
    
    if rand_pos==0:
        h_pre=hx_i
    else:
        h_pre=old_hx_discrete[rand_pos-1]

    new_hx_discrete[rand_pos]=h_pre+a
    new_action_protocol=np.diff(new_hx_discrete)
    new_action_protocol=np.concatenate(([new_hx_discrete[0]-hx_i],new_action_protocol))
    return new_action_protocol,new_hx_discrete
    

def simulate_anneal(params):
    """
    Purpose:
        Performs simulated annealing by trying to maximize the fidelity between a state evolved
        using a quantum time evolution protocol and a predefined target state.
    Return: (4-tuple)
        Number of times of fidelity evaluation,
        Best obtained fidelity,
        Corresponding action protocol,
        Corresponding hx fiels vs time slice"
    """
    
    global hx_discrete
    
    if params['verbose']:
        enablePrint()
    else:
        blockPrint()
    

    # Simulated annealing parameters
    T=params['Ti']
    Ti=T
    N_quench=params['N_quench']
    RL_constraint=params['RL_CONSTRAINT']
    step=0.0
    sweep_size=params['sweep_size']
    beta=1./T

    
    # Fidelity calculation parameters
    psi_i=params['psi_i']
    H=params['H']
    N_time_step=params['N_time_step']
    delta_t=params['delta_t']
    psi_target=params['psi_target']
    hx_i=params['hx_i']
    
    # Initializing variables
    action_protocol,hx_discrete=random_trajectory(hx_i,N_time_step)
    
    best_action_protocol=action_protocol
    best_hx_discrete=hx_discrete
    best_fid=Fidelity(psi_i,H,N_time_step,delta_t,psi_target)

    old_hx_discrete=best_hx_discrete
    old_action_protocol=best_action_protocol
    old_fid=best_fid
    
    count_fid_eval=0
    
    while T>1E-6:
        if N_quench==0:break
    
        print("Current temperature: %.4f\tBest fidelity: %.4f\tFidelity count: %i"%(T,best_fid,count_fid_eval))
        
        beta=1./T
        for _ in range(sweep_size):
            new_action_protocol,new_hx_discrete=propose_new_trajectory(old_action_protocol,old_hx_discrete,hx_i,N_time_step,RL_constraint=RL_constraint)
            hx_discrete=new_hx_discrete
            
            new_fid=Fidelity(psi_i,H,N_time_step,delta_t,psi_target)
            count_fid_eval+=1
            
            if new_fid > best_fid: # Record best encountered !
                best_fid=new_fid
                best_action_protocol=new_action_protocol
                best_hx_discrete=new_hx_discrete
            
            dF=(new_fid-old_fid)
            
            if dF>0:
                old_hx_discrete=new_hx_discrete
                old_action_protocol=new_action_protocol
                old_fid=new_fid           
            elif np.random.uniform() < np.exp(beta*dF):
                old_hx_discrete=new_hx_discrete
                old_action_protocol=new_action_protocol
                old_fid=new_fid
        
        step+=1.0
        T=Ti*(1.0-step/N_quench)
    
    print("Performing 5 sweeps (1 sweep=%i evals) at zero-temperature"%sweep_size)
    ## Need to have a stochastic gradient descent option here ... so number of sweeps is optionally calibrated 
    n_iter_without_progress=0
    propose_update=np.arange(N_time_step)
    np.random.shuffle(propose_update)
    propose_update_pos=0
    
    for _ in range(5*sweep_size): ## Perform greedy sweeps (zero-temperature):
        rand_pos=propose_update[propose_update_pos]
        new_action_protocol,new_hx_discrete=propose_new_trajectory(old_action_protocol,old_hx_discrete,
                                                                   hx_i,N_time_step,RL_constraint=RL_constraint,
                                                                   rand_pos=rand_pos
                                                                   )
        hx_discrete=new_hx_discrete
        new_fid=Fidelity(psi_i,H,N_time_step,delta_t,psi_target)
        
        count_fid_eval+=1
        n_iter_without_progress+=1
        propose_update_pos+=1
        
        if new_fid > best_fid:# Record best encountered !
            n_iter_without_progress=0
            propose_update_pos=0
            np.random.shuffle(propose_update)
            best_fid=new_fid
            best_action_protocol=new_action_protocol
            best_hx_discrete=new_hx_discrete
            
        dF=(new_fid-old_fid)
        if dF>0:
            old_hx_discrete=new_hx_discrete
            old_action_protocol=new_action_protocol
            old_fid=new_fid
        if _%10 == 0:
            '{:>8} {:>8} {:>8}'
            print('{:<25} {:<25} {:<20}'.format("Zero T iteration # %i"%_,"Best fidelity: %.4f"%best_fid,"Fidelity count: %i"%count_fid_eval))
        if n_iter_without_progress > N_time_step-1: print("Reached local minima with probability 1");break;
    print("~~ Done ~~")
    
    enablePrint()
    return count_fid_eval,best_fid,best_action_protocol,best_hx_discrete

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def check_custom_protocol(hx_protocol,J=1.236,
                          L=1,hz=1.0,hx_init_state=-1.0,hx_target_state=1.0,
                          delta_t=0.05):
    
    """ 
    Purpose:
        Just quickly checking a given protocol (or performance of fidelity function for instance)
    Return:
        Obtained fidelity
    """
    global action_set,hx_discrete
    N_time_step=len(hx_protocol)
    
    param={'J':J,'hz':hz,'hx':hx_init_state} # Hamiltonian kwargs 
    hx_discrete=[0]*N_time_step # dynamical part at every time step (initiaze to zero everywhere)
    # full system hamiltonian
    H,_ = Hamiltonian.Hamiltonian(L,fct=hx_vs_t,**param)
    # calculate initial and final states
    hx_discrete[0]=hx_init_state # just a trick to get initial state
    E_i, psi_i = H.eigsh(time=0,k=1,which='SA')
    hx_discrete[0]=hx_target_state # just a trick to get final state
    E_f, psi_target = H.eigsh(time=0,k=1,which='SA')

    print("No evolution yields dot(psi_i,psi_target)=",np.dot(psi_target.flatten(),psi_i.flatten())**2)
    
    hx_discrete=hx_protocol
    return Fidelity(psi_i,H,N_time_step,delta_t,psi_target) 
       
# Run main program !
if __name__ == "__main__":
    main()
