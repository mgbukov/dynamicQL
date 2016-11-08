import numpy as np
import pickle
import Hamiltonian_alex as Hamiltonian
from quspin.operators import exp_op
import time
import math
from scipy.sparse.linalg import expm_multiply as expm

np.random.seed(0)                
def main():
    
    global action_set,hx_discrete
    
    L = 1 # system size
    J = 1.0/0.809 # zz interaction
    hz = 0.5 #0.9045/0.809 #1.0 # hz field
    hx_i = 0.# -1.0 # initial hx coupling
    hx_f = 2.0 #+1.0 # final hx coupling
    N_time_step=40
    delta_t=0.05
    N_restart=10
    
    param={'J':J,'hz':hz,'hx':hx_i}
    
    # dynamical part at every time step (initiaze to zero everywhere) 
    hx_discrete=[0]*N_time_step
    
    # full system hamiltonian
    H,_ = Hamiltonian.Hamiltonian(L,fct=hx_vs_t,**param)
   # print(H)
    
    # calculate initial and final states
    hx_discrete[0]=hx_i # just a trick to get initial state
    E_i, psi_i = H.eigsh(time=0,k=1,which='SA')
    hx_discrete[0]=hx_f # just a trick to get final state
    E_f, psi_target = H.eigsh(time=0,k=1,which='SA')
    hx_discrete[0]=0
    
    action_set=[0.0,0.02,0.05,0.08,0.1,0.2,0.4,0.8,-0.02,-0.05,-0.08,-0.1,-0.2,-0.4,-0.8]
    #action_set=[-0.2,0.,0.2]
    
    
    param_SA={'Ti':1,'dT':0.01,'sweep_size':40,
              'psi_i':psi_i,'H':H,'N_time_step':N_time_step,
              'delta_t':delta_t,'psi_target':psi_target,
              'hx_i':hx_i}
    
    best_result=[0,0,0]
    all_results=[]
    for it in range(N_restart):
        print("Iteration:",it)
        result=simulate_anneal(param_SA)
        all_results.append(result)
        print("Best fidelity during iteration: %s"%result[0])
        print("Corresponding trajectory:",result[2])
        
        if result[0] > best_result[0]:
            best_result=result
    
    
    print("Best of all:",best_result)
    print("All results:",all_results)
    
    
    #Saving results:
    pkl_file=open('data/allresultsL1.pkl','wb')
    pickle.dump(all_results,pkl_file)
    pkl_file.close()
    
    
def Fidelity(psi_i,H,N_time_step,delta_t,psi_target):
    """
    Calculates final fidelity by evolving psi_i over a N_time_step 
    Returns the overlap between the target state psi_target and the evolved state
    
    """    
    psi_evolve=psi_i.copy()
    for t in range(N_time_step):
        psi_evolve = exp_op(H(time=t),a=-1j*delta_t).dot(psi_evolve)
    
    #print(psi_evolve)
    return abs(np.sum(np.conj(psi_evolve)*psi_target))**2

def Fidelity_fast(psi_i,H,N_time_step,delta_t,psi_target):
    # Useful for small systems size)
    psi_evolve=psi_i.copy()
    for t in range(N_time_step):
       psi_evolve=expm(-1j*delta_t*(H(time=t).todense()),psi_evolve)
    return abs(np.dot(np.transpose(np.conj(psi_evolve)),psi_target)[0,0])**2

   
def hx_vs_t(time): return hx_discrete[int(time)]

def random_trajectory(hx_i,N_time_step):
    '''
    Returns the action protocol and the corresponding trajectory
    '''
    
    action_protocol=np.random.choice(action_set,N_time_step)
    return action_protocol,hx_i+np.cumsum(action_protocol)

def propose_new_trajectory(old_action_protocol,hx_i,N_time_step):
    '''
    Given the old_action_protocol, makes a random change and returns the new action protocol
    '''
    new_action_protocol=np.copy(old_action_protocol)
    rand_pos=np.random.randint(N_time_step)
    #print(rand_pos)
    new_action_protocol[rand_pos]=np.random.choice(action_set)
    #print(N_tim)
    return new_action_protocol,hx_i+np.cumsum(new_action_protocol)

def simulate_anneal(params):
    
    global hx_discrete
    
    # Simulated annealing parameters
    T=params['Ti']
    dT=params['dT']
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
    
    while T>0.:
        print("Current temperature=%s"%T,"Best fidelity=%s"%best_fid)
        #print("Current temperature=%s"%(1./beta),"Best fidelity=%s"%best_fid)
        beta=1./T
        for _ in range(sweep_size):
            new_action_protocol,new_hx_discrete=propose_new_trajectory(old_action_protocol,hx_i,N_time_step)
            hx_discrete=new_hx_discrete
            start=time.time()
            new_fid=Fidelity(psi_i,H,N_time_step,delta_t,psi_target)
            #print(time.time()-start)
            #print(new_fid)
            
            if new_fid > best_fid:
                # Record best encountered !
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
            #else:
            #    print("move rejected!",np.exp(beta*dF))
        

        T-=dT
      
    return best_fid,best_action_protocol,best_hx_discrete

# Run main program !
if __name__ == "__main__":
    main()