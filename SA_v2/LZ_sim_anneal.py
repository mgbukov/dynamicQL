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
    
    with open("ES_L-06_T-0.500_n_step-28.pkl", ‘rb’) as f:
	    fidelities=pickle.load(f)
    nfid=fidelities.shape[0]

    fid_and_energy=np.empty((nfid,2),dtype=np.float)
    for i,f in zip(range(nfid),fidelity):
        model.update_protocol(b2_array(i), w = 28)
        fid_and_energy[i][0]=fid
        fid_and_energy[i][1]=model.compute_energy()

    with open("ES_L-06_T-0.500_n_step-28-test.pkl", ‘wb’) as f:
	    fidelities=pickle.dump(fid_and_energy,f)
    
    #fid = model.compute_fidelity()
    #energy = model.compute_energy()
    #print(model.compute_energy())
    #print(model.compute_energy(model.psi_target))
    #print(energy)

    exit()
    model.anneal()
        
    if outfile_name=="auto": outfile_name=ut.make_file_name(param_SA)
    
    to_save_par=['Ti','psi_i','N_time_step',
                'delta_t','psi_target','hx_i','N_quench','RL_CONSTRAINT',
                'hx_initial_state','hx_final_state','L','J','hz','action_set',
                'symmetrize'
    ]
    
    file_content=ut.read_current_results('data/%s'%outfile_name)
    
    # Read current data if it exists
    if file_content :
        dict_to_save_parameters, all_results = file_content
        N_current_restart = len(all_results)
        print("Data with %i samples available !" % N_current_restart) 
    else :
        dict_to_save_parameters = dict(zip(to_save_par,[param_SA[p] for p in to_save_par]))
        all_results=[]
        N_current_restart = 0
    
    #print(N_current_restart," ",N_restart)
    for it in range(N_current_restart, N_restart):
        print("\n\n-----------> Starting new iteration <-----------")
        start_time=time.time()
    
        count_fid_eval,best_fid,best_action_protocol,best_hx_discrete = simulate_anneal(param_SA)
    
        result=[count_fid_eval,best_fid,best_action_protocol,best_hx_discrete]
        print("\n----------> RESULT FOR ANNEALING NO %i <-------------"%(it+1))
        print("Number of fidelity eval \t%i"%count_fid_eval)
        print("Best fidelity \t\t\t%.4f"%best_fid)
        print("Best hx_protocol\t\t",list(best_hx_discrete))

        if L > 1:  
            _,E,delta_E,Sd,Sent = MB_observables(best_hx_discrete, param_SA, matrix_dict, fin_vals=True)
            result = result + [E, delta_E, Sd, Sent] # Appending Energy, Energy fluctuations, Diag. entropy, Ent. entropy
        
        all_results.append(result)
        with open('data/%s'%outfile_name,'wb') as pkl_file:
            ## Here read first then save, stop if reached quota
            pickle.dump([dict_to_save_parameters,all_results],pkl_file);pkl_file.close()
            
        print("Saved iteration --> %i to %s"%(it,'data/%s'%outfile_name))
        print("Iteration run time --> %.2f s"%(time.time()-start_time))
    
    print("\n Thank you and goodbye !")
    
    
def fast_Fidelity(psi_i,H,N_time_step,delta_t,psi_target):
    """
    Purpose:
        Calculates final fidelity by evolving psi_i over a N_time_step using the matrix dictionary 
    Return: 
        Norm squared between the target state psi_target and the evolved state (according to the full hx_discrete protocol)
    """   
    
    psi_evolve=psi_i.copy()
    #print(np.shape(psi_evolve))
    #print(np.shape(matrix_dict[-4.]))
    for t in range(N_time_step):
        psi_evolve = matrix_dict[hx_discrete[t]].dot(psi_evolve)

    return abs(np.sum(np.conj(psi_evolve)*psi_target))**2


# Returns the hx field at a given time slice 
def hx_vs_t(time): return hx_discrete[int(time)]

def random_trajectory(hx_i, N_time_step, symmetrize=False):
    '''
    Purpose:
        Generates a random trajectory
    Return:
        Action protocol,hx_discrete ; action taken at every time slice, field at every time slice
    '''
    new_action_protocol=[]
    current_h=hx_i
    for _ in range(N_time_step):    
        while True:
            action_choice=np.random.choice(action_set)
            current_h+=action_choice
            if abs(current_h) < hx_max+0.0001:
                new_action_protocol.append(action_choice)
                break
            else:
                current_h-=action_choice

    new_hx_discrete = hx_i+np.cumsum(new_action_protocol)

    if symmetrize:
        symmetrize_protocol(new_hx_discrete)
        new_action_protocol=np.diff(new_hx_discrete)
        new_action_protocol=np.concatenate(([new_hx_discrete[0]-hx_i],new_action_protocol))

    return new_action_protocol, new_hx_discrete

# Check if two floats are equal according to some precision
def are_equal(a,b,prec=0.000001):
    return abs(a-b)<prec

# Check if a is in set by comparing floats (with a given precision)
def is_element_of_set(a,set,prec=0.000001):
    return np.min(abs(set-a)) < prec

def avail_action_RL(time_position,old_action_protocol,old_hx_discrete,hx_i): # simulate SA in the space of actions, not field... need to recompute traj. 
    """
    Purpose:
        Get available actions at a specific time slice given a protocol and with RL constratins 
    Return:
        A list of available actions
    """
    action_subset=[]
    N_time_step=len(old_hx_discrete)

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

def avail_action(time_position,old_action_protocol,old_hx_discrete,hx_i):
    """
    Purpose:
        Get available actions at a specific time slice given a protocol without any constraints (except having abs(hx)<abs(hx_max) 
    Return:
        A list of available actions
    """
    action_subset=[]
    N_time_step=len(old_hx_discrete)
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

def propose_new_trajectory(old_action_protocol,old_hx_discrete,hx_i,N_time_step,RL_constraint=False,rand_pos=None,symmetrize=False):
    '''
    Purpose:
        Given the old_action_protocol, makes a random change and returns the new action protocol
        
    Return:
        New action protocol,New hx as a function of the time slice
    '''
    N_time_random = N_time_step

    new_hx_discrete=np.copy(old_hx_discrete)

    if symmetrize:
        assert (N_time_step % 2) == 0, "Works only for even # of steps"
        N_time_random=int(N_time_step/2)

    if rand_pos==None:
        rand_pos = np.random.randint(N_time_random)
    current_action = old_action_protocol[rand_pos]
    
    tmp=[]
    count=0

    while True:
        if RL_constraint:
            aset=avail_action_RL(rand_pos,old_action_protocol,old_hx_discrete,hx_i)
        else:
            aset=avail_action(rand_pos,old_action_protocol,old_hx_discrete,hx_i)
        for aa in aset:
            if not are_equal(aa,current_action):
                tmp.append(aa)
    
        if len(tmp)==0: 
            rand_pos=np.random.randint(N_time_random) 
            current_action=old_action_protocol[rand_pos]
        else:
            a=np.random.choice(tmp)
            break
    
    if rand_pos==0:
        h_pre=hx_i
    else:
        h_pre=old_hx_discrete[rand_pos-1]

    new_hx_discrete[rand_pos]=h_pre+a
    if symmetrize:
        symmetrize_protocol(new_hx_discrete)

    new_action_protocol=np.diff(new_hx_discrete)
    new_action_protocol=np.concatenate(([new_hx_discrete[0]-hx_i],new_action_protocol))

    return new_action_protocol,new_hx_discrete


def symmetrize_protocol(hx_protocol):
    Nstep=len(hx_protocol)
    half_N=int(Nstep/2)
    for i in range(half_N):
        hx_protocol[-(i+1)]=-hx_protocol[i]

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
    
    enablePrint() if params['verbose'] else blockPrint()
    
    # Simulated annealing parameters
    T = params['Ti']
    Ti = T
    eta = 1e-14
    N_quench = params['N_quench']

    RL_constraint = params['RL_CONSTRAINT']
    step = 0.0
    beta = 1./T
    option_fidelity = ('fast' if params['fidelity_fast'] else 'standard')

    # Fidelity calculation parametersls
    psi_i = params['psi_i']
    H = params['H']
    N_time_step = params['N_time_step']
    delta_t = params['delta_t']
    psi_target = params['psi_target']
    hx_i = params['hx_i']
    symmetrize = params['symmetrize']
    
    max_iter_zero_T = 20 * N_time_step * len(action_set) # very-high probability of not reaching this threshold ...

    # Initializing variables
    action_protocol,hx_discrete=random_trajectory(hx_i,N_time_step,symmetrize=symmetrize)
    
    best_action_protocol = action_protocol
    best_hx_discrete = hx_discrete
    best_fid = Fidelity(psi_i,H,N_time_step,delta_t,psi_target,option=option_fidelity)
    
    old_hx_discrete = best_hx_discrete
    old_action_protocol = best_action_protocol
    old_fid = best_fid
    
    count_fid_eval=0

    if N_quench > 0 : 
        print("[SA] : Quenching to zero temperature in %i steps"%N_quench)

    while T>1E-6:
        # Simulated annealing
        if N_quench == 0:
            break
    
        beta = 1./T

        new_action_protocol,new_hx_discrete=propose_new_trajectory(old_action_protocol,old_hx_discrete,hx_i,N_time_step,
                                                                        RL_constraint=RL_constraint,symmetrize=symmetrize)
        hx_discrete = new_hx_discrete
            
        new_fid = Fidelity(psi_i,H,N_time_step,delta_t,psi_target,option=option_fidelity)
        count_fid_eval+=1
            
        if new_fid > best_fid: # Record best encountered !
            print("Current temperature: %.5f\tBest fidelity: %.6f\tCurrent fidelity:%.6f\tFidelity count: %i"%(T, best_fid, old_fid, count_fid_eval))
            best_fid=new_fid
            best_action_protocol=new_action_protocol
            best_hx_discrete=new_hx_discrete
            
        dF = (new_fid-old_fid)
            
        if dF > 0:
            old_hx_discrete=new_hx_discrete
            old_action_protocol=new_action_protocol
            old_fid=new_fid           
        elif np.random.uniform() < np.exp(beta*dF):
            old_hx_discrete=new_hx_discrete
            old_action_protocol=new_action_protocol
            old_fid=new_fid
        
        step += 1.0
        T = Ti * (1.0-step/N_quench)
    
    print("\n[SD] : Performing zero-temperature annealing")
    
    n_iter_without_progress = 0
    propose_update=np.arange(N_time_step)
    np.random.shuffle(propose_update)
    propose_update_pos = 0
    
    for _ in range(max_iter_zero_T):
        # Stochastic descent
        
        rand_pos=propose_update[propose_update_pos]
        new_action_protocol,new_hx_discrete=propose_new_trajectory(old_action_protocol, old_hx_discrete,
                                                                   hx_i,N_time_step, RL_constraint=RL_constraint,
                                                                   rand_pos=rand_pos, symmetrize=symmetrize)
        hx_discrete=new_hx_discrete
        new_fid=Fidelity(psi_i, H, N_time_step, delta_t, psi_target, option=option_fidelity)
        
        count_fid_eval += 1
        n_iter_without_progress += 1
        propose_update_pos += 1
        
        dF = new_fid - old_fid
        if ( abs(dF) > eta ) & (dF > 0) :

            print("Current temperature: %.5f\tBest fidelity: %.6f\tCurrent fidelity:%.6f\tFidelity count: %i"%(T, best_fid, old_fid, count_fid_eval))
            
            n_iter_without_progress = 0
            propose_update_pos = 0
            np.random.shuffle(propose_update)
            best_fid = new_fid
            best_action_protocol = new_action_protocol
            best_hx_discrete = new_hx_discrete
            
            old_hx_discrete = new_hx_discrete
            old_action_protocol = new_action_protocol
            old_fid = new_fid

        if n_iter_without_progress > N_time_step-1: 
                break

    print("=====> DONE <=====")
    
    enablePrint()
    return count_fid_eval,best_fid,best_action_protocol,best_hx_discrete

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

class custom_protocol():
    def __init__(self,J=-1.0,L=1,hz=1.0,hx_init_state=-1.0,hx_target_state=1.0,
                    delta_t=0.05,hx_i=-4.,hx_max=4.,action_set_=[-8.,0.,8.],
                    option='standard'):
        
        global action_set,hx_discrete
        action_set=action_set_
        #N_time_step=len(hx_protocol)
        
        self.option=option
        self.delta_t=delta_t
        param={'J':J,'hz':hz,'hx':hx_init_state} # Hamiltonian kwargs 
        param_SA={'J':J, 'hz': hz, 'L': L, 'delta_t': delta_t, 'hx_final_state':hx_target_state,
        'action_set':action_set_,'hx_i':hx_i
        }
        hx_discrete=[0] # dynamical part at every time step (initiaze to zero everywhere)
        # full system hamiltonian
        self.H,_ = Hamiltonian.Hamiltonian(L,fct=hx_vs_t,**param)
        
        # calculate initial and final states
        hx_discrete[0]=hx_init_state # just a trick to get initial state
        _, self.psi_i = self.H.eigsh(time=0,k=1,which='SA')
        hx_discrete[0]=hx_target_state # just a trick to get final state
        _, self.psi_target = self.H.eigsh(time=0,k=1,which='SA')

        param_SA['hx_target_state']=self.psi_target

        print("--> Overlap between initial and target state %.4f"%(abs(np.sum(np.conj(self.psi_i)*self.psi_target))**2))
        
        if option is 'fast':
            h_set=compute_h_set(hx_i, hx_max)
            precompute_expmatrix(param_SA, h_set, self.H)
        
    def evaluate_protocol_fidelity(self,hx_protocol): 
        global hx_discrete       
        hx_discrete=hx_protocol
        N_time_step=len(hx_discrete)
        if self.option is 'standard':
            return Fidelity(self.psi_i,self.H,N_time_step,self.delta_t,self.psi_target,option='standard')
        elif self.option is 'fast':
            return Fidelity(self.psi_i,self.H,N_time_step,self.delta_t,self.psi_target,option='fast')    
        else:
            assert False,'Wrong option, use either fast or standard'
    
    def compute_eig(self,hx):
        global hx_discrete  
        hx_discrete[0]=hx # just a trick to get initial state
        E_i, psi_i = self.H.eigsh(time=0,k=1,which='SA')
        return E_i, psi_i
    
# Run main program !
if __name__ == "__main__":
    main()
