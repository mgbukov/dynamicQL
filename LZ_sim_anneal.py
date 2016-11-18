import numpy as np
import pickle
import Hamiltonian_alex as Hamiltonian
from quspin.operators import exp_op
import time
import math
import sys # for running in batch from terminal
from scipy.sparse.linalg import expm_multiply as expm
#from matplotlib import pyplot as plt
                
def main():
    
    global action_set,hx_discrete,max_step
    
    L = 1 # system size
    J = 1.0/0.809 # zz interaction
    hz = 0.5 #0.9045/0.809 #1.0 # hz field
    hx_i = -1.0# -1.0 # initial hx coupling
    hx_f = 1.0 #+1.0 # final hx coupling
    N_quench=100
    delta_t=0.05
    N_restart=50
    
    action_set1=[-2.0,0.,2.0]
    action_set2=[0.0,-0.01,0.01,0.02,-0.02,0.04,-0.04,-0.08,0.08,-0.16,0.16,-0.32,0.32,-0.64,0.64,-1.28,1.28]
    action_set3=[0.,0.02,0.05,0.08,0.1,0.2,0.4,0.8]
    action_set4=[-10.,0.,10.]
    
    if len(sys.argv)>1:
        # argv order : Number of time step, action set number, filename for output
        N_time_step=int(sys.argv[1])
        action_set_no=sys.argv[2]
        action_set=eval('action_set'+action_set_no)
        outfile_name=sys.argv[3]
    else:
        N_time_step=40
        outfile_name='BB_action_set_1'
        action_set=action_set1
        max_step=max(action_set)

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
    
    print("Current action set:",action_set)
    #===========================================================================
    # 
    # hx_discrete=[ 1.,1.,1.,1.,1.,-1.,-1.,-1.,-1.,-1.]
    # print(Fidelity(psi_i,H,N_time_step,delta_t,psi_target))
    # test=np.loadtxt("data/text.dat",delimiter="\t")
    # fbest=0.0
    # i=0
    # 
    # for hx_tmp in test:
    #     print(i)
    #     hx_discrete=hx_tmp
    #     f=Fidelity(psi_i,H,N_time_step,delta_t,psi_target)
    #     if f>fbest:
    #         fbest=f
    #         hbest=hx_discrete
    #     i+=1
    # print(fbest)
    # print(hbest)
    # exit()
    # #test=np.loadtxt("data/text.dat",delimiter="\t")
    #===========================================================================
    
    param_SA={'Ti':0.04,'sweep_size':40,
                'psi_i':psi_i,'H':H,'N_time_step':N_time_step,
                'delta_t':delta_t,'psi_target':psi_target,
                'hx_i':hx_i,'N_quench':N_quench}
    

    
    all_results=[]
    for it in range(N_restart):
       # print("Iteration:",it)
        result=simulate_anneal(param_SA)
        all_results.append(result)
        print(result[0])
        print(result[1])
        print(result[2])
        #print("Best fidelity during iteration: %s"%result[0])
        #print("Corresponding trajectory:",result[2])
        
        #if result[0] > best_result[0]:
        #    best_result=result
    
    #print("Best of all:",best_result)
    #print("All results:",all_results)
    
    
    #Saving results:
    pkl_file=open('data/%s.pkl'%outfile_name,'wb')
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
    action_protocol=[]
    current_h=hx_i
    for _ in range(N_time_step):    
        while True:
            action_choice=np.random.choice(action_set)
            current_h+=action_choice
            if abs(current_h) < 2.0001:
                action_protocol.append(action_choice)
                break
            else:
                current_h-=action_choice
    
    #action_protocol=np.random.choice(action_set,N_time_step)
    return action_protocol,hx_i+np.cumsum(action_protocol)

def propose_new_trajectory(old_action_protocol,old_hx_discrete,hx_i,N_time_step):
    '''
    Given the old_action_protocol, makes a random change and returns the new action protocol
    '''
    new_hx_discrete=np.copy(old_hx_discrete)
    rand_pos=np.random.randint(N_time_step)
    aset=np.array(action_set)
    count=0
    while True:
        count+=1
        a=np.random.choice(action_set)
        if abs(a) > 1e-6:
            r=new_hx_discrete[rand_pos]+a
            if rand_pos != 0 and rand_pos != N_time_step-1:
                dhpre=np.min(np.abs(aset-abs(r-new_hx_discrete[rand_pos-1])))
                dhpost=np.min(np.abs(aset-abs(r-new_hx_discrete[rand_pos+1])))
                
                if abs(r) < 2.0001 and dhpre < 0.00001 and dhpost < 0.00001:
                    new_hx_discrete[rand_pos]+=a
                    break
            elif rand_pos == 0:
                dhpre=np.min(np.abs(aset-abs(r-hx_i)))
                dhpost=np.min(np.abs(aset-abs(r-new_hx_discrete[rand_pos+1])))
                if abs(r) < 2.0001 and dhpre < 0.00001 and dhpost < 0.00001:
                    new_hx_discrete[rand_pos]+=a
                    break
            else:
                dhpre=np.min(np.abs(aset-abs(r-new_hx_discrete[rand_pos-1])))
                if abs(r) < 2.0001 and dhpre < 0.00001:
                    new_hx_discrete[rand_pos]+=a
                    break
                
        if count > 10:
            rand_pos=np.random.randint(N_time_step)
    #new_action_protocol=np.copy(old_action_protocol)
    new_action_protocol=np.diff(new_hx_discrete)
    new_action_protocol=np.concatenate(([new_hx_discrete[0]-(-1.0)],new_action_protocol))
    return new_action_protocol,new_hx_discrete
    #new_action_protocol[rand_pos]=np.random.choice(action_set)
    #print(N_tim)
    #return new_action_protocol,hx_t
    

def simulate_anneal(params):
    
    global hx_discrete
    
    # Simulated annealing parameters
    T=params['Ti']
    Ti=T
    N_quench=params['N_quench']
    step=0.0
    #dT=params['dT']
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
    
    while T>1E-6:
        #print(T,best_fid)
        print("Current temperature=%s"%T,"Best fidelity=%s"%best_fid)
        #print("Current temperature=%s"%(1./beta),"Best fidelity=%s"%best_fid)
        beta=1./T
        for _ in range(sweep_size):
            new_action_protocol,new_hx_discrete=propose_new_trajectory(old_action_protocol,old_hx_discrete,hx_i,N_time_step)
            hx_discrete=new_hx_discrete
            
            new_fid=Fidelity(psi_i,H,N_time_step,delta_t,psi_target)
            
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
            #else:
            #    print("move rejected!",np.exp(beta*dF))
        
        step+=1.0
        T=Ti*(1.0-step/N_quench)
      
    for _ in range(10*sweep_size): ## Perform greedy sweeps (zero-temperature):
            new_action_protocol,new_hx_discrete=propose_new_trajectory(old_action_protocol,old_hx_discrete,hx_i,N_time_step)
            hx_discrete=new_hx_discrete
            new_fid=Fidelity(psi_i,H,N_time_step,delta_t,psi_target)
            if new_fid > best_fid:# Record best encountered !
                best_fid=new_fid
                best_action_protocol=new_action_protocol
                best_hx_discrete=new_hx_discrete
            
            dF=(new_fid-old_fid)
            if dF>0:
                old_hx_discrete=new_hx_discrete
                old_action_protocol=new_action_protocol
                old_fid=new_fid
    
    return best_fid,best_action_protocol,best_hx_discrete

# Run main program !
if __name__ == "__main__":
    main()
