import numpy as np
import pickle
import Hamiltonian_alex as Hamiltonian
from quspin.operators import exp_op
import time
import math
import sys # for running in batch from terminal
from scipy.sparse.linalg import expm_multiply as expm
#from matplotlib import pyplot as plt
      
np.set_printoptions(precision=4)
          
def main():
    
    global action_set,hx_discrete,hx_max,FIX_NUMBER_FID_EVAL
    
    L = 1 # system size
    J = 1.0/0.809 # zz interaction
    hz = 1.0 #0.9045/0.809 #1.0 # hz field
    hx_i = -4.0# -1.0 # initial hx coupling
    hx_initial_state= -1.0 # initial state
    hx_f = 1.0 #+1.0 # final hx coupling
    N_quench=30
    delta_t=0.05
    N_restart=10
    hx_max=4
    max_fid_eval=1000
    FIX_NUMBER_FID_EVAL=False
    
    action_set1=[-8.0,0.,8.]
    action_set2=np.array([0.01,0.05,0.1,0.2,0.5,1.,2.,3.,4.],dtype=np.float32) # 17 actions in total
    action_set2=list(np.round(np.concatenate((action_set2,-1.0*action_set2,[0])),3))
    action_set3=[8.0,-8.0,0.0]
    #action_set4=[-0.]
    
    if len(sys.argv)>1:
        # argv order : Number of time step, action set number, filename for output max_fid_eval
        # 
        N_time_step=int(sys.argv[1])
        action_set_no=sys.argv[2]
        action_set=eval('action_set'+action_set_no)
        outfile_name=sys.argv[3]
        max_fid_eval=int(sys.argv[4])
    else:
        N_time_step=20
        outfile_name='BB_action_set_1'
        action_set=action_set1
        max_fid_eval=3000

    param={'J':J,'hz':hz,'hx':hx_i}
    
    # dynamical part at every time step (initiaze to zero everywhere) 
    hx_discrete=[0]*N_time_step
    
    # full system hamiltonian
    H,_ = Hamiltonian.Hamiltonian(L,fct=hx_vs_t,**param)
   # print(H)
    
    # calculate initial and final states
    hx_discrete[0]=hx_initial_state # just a trick to get initial state
    E_i, psi_i = H.eigsh(time=0,k=1,which='SA')
    hx_discrete[0]=hx_f # just a trick to get final state
    E_f, psi_target = H.eigsh(time=0,k=1,which='SA')
    hx_discrete[0]=0
    
    print("Current action set:",action_set)
    #hx_discrete=[-1.05, -0.050000000000000044, 0.95, 1.95, 1.95, 1.9, 1.9, 1.9, 1.9, 0.8999999999999999, -0.10000000000000009, -1.1, -1.3, -1.4000000000000001, -1.3800000000000001, -1.4800000000000002, -1.9800000000000002, -1.9900000000000002, -2.0, -2.0]
    #N_time_step=len(hx_discrete)
    #hx_discrete=hx_discrete[:-1]
    #print(len(hx_discrete))
    #print(Fidelity(psi_i,H,N_time_step,delta_t,psi_target))
    #print(N_time_step)
    #exit()
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
    sweep_size=N_time_step*len(action_set)
    if FIX_NUMBER_FID_EVAL:
        N_quench=(max_fid_eval-5*sweep_size)//sweep_size
        assert N_quench >= 0
    print("Using N_quench=:%d"%N_quench)
    
    param_SA={'Ti':0.04,'sweep_size':sweep_size,
                'psi_i':psi_i,'H':H,'N_time_step':N_time_step,
                'delta_t':delta_t,'psi_target':psi_target,
                'hx_i':hx_i,'N_quench':N_quench
                }

    
    all_results=[]
    for it in range(N_restart):
       # print("Iteration:",it)
        result=simulate_anneal(param_SA)
        print(result)
        all_results.append(result)
        pkl_file=open('data/%s.pkl'%outfile_name,'wb')
        pickle.dump(all_results,pkl_file)
        pkl_file.close()
        #print(result[0])
        #print(result[1])
        #print(result[2])
        #np.savetxt("test.txt",result[1])
        #print("Best fidelity during iteration: %s"%result[0])
        #print("Corresponding trajectory:",result[2])
        
        #if result[0] > best_result[0]:
        #    best_result=result
    
    #print("Best of all:",best_result)
    #print("All results:",all_results)
    
    
    #Saving results:
#pkl_file=open('data/%s.pkl'%outfile_name,'wb')
#pickle.dump(all_results,pkl_file)
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
            if abs(current_h) < hx_max+0.0001:
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
    option=0
    # w/o RL constraints
    if option == 0: 
        while True:
            a=np.random.choice(action_set)
            if abs(a) > 1e-6:
                r=new_hx_discrete[rand_pos]+a
                if abs(r) < hx_max+0.00001:
                    new_hx_discrete[rand_pos]+=a
                    break
    else:
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
    new_action_protocol=np.concatenate(([new_hx_discrete[0]-hx_i],new_action_protocol))
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
    #max_fid_eval=params['max_fid_eval']
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
        #print(T,best_fid)
        print("Current temperature: %.4f\tBest fidelity: %.4f\tFidelity count: %i"%(T,best_fid,count_fid_eval))
        
        #print("Current temperature=%s"%(1./beta),"Best fidelity=%s"%best_fid)
        beta=1./T
        for _ in range(sweep_size):
            new_action_protocol,new_hx_discrete=propose_new_trajectory(old_action_protocol,old_hx_discrete,hx_i,N_time_step)
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
            #else:
            #    print("move rejected!",np.exp(beta*dF))
        
        step+=1.0
        T=Ti*(1.0-step/N_quench)
      
    for _ in range(5*sweep_size): ## Perform greedy sweeps (zero-temperature):
        #print(_)
        new_action_protocol,new_hx_discrete=propose_new_trajectory(old_action_protocol,old_hx_discrete,hx_i,N_time_step)
        hx_discrete=new_hx_discrete
        new_fid=Fidelity(psi_i,H,N_time_step,delta_t,psi_target)
        count_fid_eval+=1
        if new_fid > best_fid:# Record best encountered !
            best_fid=new_fid
            best_action_protocol=new_action_protocol
            best_hx_discrete=new_hx_discrete
            
        dF=(new_fid-old_fid)
        if dF>0:
            old_hx_discrete=new_hx_discrete
            old_action_protocol=new_action_protocol
            old_fid=new_fid
    print("Done")
    return count_fid_eval,best_fid,best_action_protocol,best_hx_discrete

# Run main program !
if __name__ == "__main__":
    main()