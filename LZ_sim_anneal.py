import numpy as np

#print("erhere")


import Hamiltonian
import numpy as np
import time
#from builtins import False
#import deepcopy


def Fidelity(psi_i,H_fid,t_vals,basis=None,psi_f=None,all_obs=False):
    """ This function calculates the physical quantities given a time-dep Hamiltonian H_fid.
        If psi_f is not given, then it returns the fidelity in the instantaneous eigenstate, otherwise
        --- the fidelity in the final state. """
    # get sysstem size
    L = basis.L 
    # evolve state
    psi_t = H_fid.evolve(psi_i,t_vals[0],t_vals,iterate=True,atol=1E-12,rtol=1E-12)

    # fidelity
    fid = []
    if all_obs:
        # entanglement entropy density
        Sent=[]
        subsys = [j for j in range(L/2)]
        # energy
        E=[]
        # energy fluctuations
        dE=[]

    for i, psi in enumerate(psi_t):

        if psi_f is not None:
            # calculate w.r.t. final state
            psi_target = psi_f
        else:
            # calculate w.r.t. instantaneous state
            _,psi_inst = H_fid.eigsh(time=t_vals[i],k=1,sigma=-100.0)
            psi_target = psi_inst.squeeze()

        fid.append( abs(psi.conj().dot(psi_target))**2 )

        if all_obs:
            E.append( H_fid.matrix_ele(psi,psi,time=t_vals[i]).real/L )

            dE.append( np.sqrt( (H_fid*H_fid(time=t_vals[i])).matrix_ele(psi,psi,time=t_vals[i]).real/L**2 - E[i]**2)  )
            #print  H_fid2.matrix_ele(psi,psi,time=t_vals[i]).real/L**2 - E[i]**2

            Sent.append( ent_entropy(psi,basis,chain_subsys=subsys)['Sent'] )

    if not all_obs:
        return fid
    else:
        return fid , E, dE, Sent

L=2

# define model params
if L==1:
    J = 0.0
else:
    J = 1.0 # zz interaction
hz = 0.5 #0.9045/0.809 #1.0 # hz field

hx_i = 0.0#-1.0 # initial hx coupling
hx_f = 1.0 #+1.0 # final hx coupling

# define ED Hamiltonian H(t)
m=0.0
b=hx_i
lin_fun = lambda t: m*t + b

#def lin_fun(t):
#    print p_vals
#    return np.interp(t, t_vals, p_vals)
# define Hamiltonian

H, basis = Hamiltonian.Hamiltonian(L,fun=lin_fun,**{'J':J,'hz':hz})
# defien Hamiltonian for interpolated fidelity
#global t_vals,p_vals
#protocol_fun = lambda t: np.interp(t, t_vals, p_vals)

#print(H.todense())
# calculate initial state
# calculate initial state
b = hx_i
E_i, psi_i = H.eigsh(time=0,k=2,which='BE',maxiter=1E10,return_eigenvectors=True)
E_i = E_i[0]
psi_i = psi_i[:,0]


# calculate final state
b = hx_f
E_f, psi_f = H.eigsh(time=0,k=2,which='BE',maxiter=1E10,return_eigenvectors=True)
E_f = E_f[0]
psi_f = psi_f[:,0]

t_vals, p_vals = [0.0,1.0,2.0], [0.0,0.2,0.4]
protocol_fun = lambda t: np.interp(t, t_vals, p_vals)
H_fid,_ = Hamiltonian.Hamiltonian(L,fun=protocol_fun,**{'J':J,'hz':hz})


#===============================================================================
# print(Fidelity(psi_i,H_fid,[t_vals[0],t_vals[-1]],basis=basis,psi_f=psi_f,all_obs=False)[-1])
# 
# p_vals=[0.0,0.2,0.8]
# 
# print(Fidelity(psi_i,H_fid,[t_vals[0],t_vals[-1]],basis=basis,psi_f=psi_f,all_obs=False)[-1])
# 
# #### ----------- QUESTIONS ------------ #######:
# # - why is the shape 78 ?
# # - why is the fidelity so high ?
# # ---> is the fidelity correctly computed ?
# 
# ####### ----------------------------
# print(H.Ns)
# print(abs(psi_i.conj().dot(psi_f))**2)
# print(psi_i)
# exit()
#===============================================================================


def random_trajectory(action_set,hi,episode_length):
    '''
    
    Returns the action protocol and the corresponding trajectory
    
    '''
    
    action_protocol=np.random.choice(action_set,episode_length)
    return action_protocol,np.insert(hi+np.cumsum(action_protocol),0,hi)

def propose_new_trajectory(current_action_protocol,action_set,hi,episode_length):
    new_action_protocol=np.copy(current_action_protocol)
    rand_pos=np.random.randint(episode_length)
    new_action_protocol[rand_pos]=np.random.choice(action_set)
    
    return new_action_protocol,np.insert(hi+np.cumsum(new_action_protocol),0,hi)


#===============================================================================
# action_set=[-0.2,0.,0.2]
# hi=-1.
# episode_length=6
# 
# ap,traj=random_trajectory([-0.2,0.,0.2],-1.,6)
# print("Initial action protocol",ap)
# print("Initial trajectory",traj)
# 
# ap_new,traj_new=propose_new_trajectory(ap,action_set,hi,episode_length)
# 
# print("New action protocol",ap_new)
# print("New trajectory",traj_new)
#===============================================================================
def simulate_anneal(Ti,dT,episode_length,action_set):
    T=Ti
    N=1000
    hi=-1.
     
    action_protocol,p_vals=random_trajectory(action_set,hi,episode_length)
    t_vals=range(episode_length+1)
    best_fid=0.0
    best_protocol=[]
#    print(action_protocol)
#    print(p_vals)
#    print(t_vals)
    #exit()
     
    #current_state=
     
    while T>0.:
        print("temp:",T)
        beta=1./T
        ti=time.time()
        
        for i in range(20):
            print(time.time()-ti)
            ti=time.time()
            
            #ti=time.time()
            action_protocol_new,p_vals_new=propose_new_trajectory(action_protocol,action_set,hi,episode_length)
            #print(time.time()-ti)
            #ti=time.time()
            
            current_fid=Fidelity(psi_i,H_fid,[t_vals[0],t_vals[-1]],basis=basis,psi_f=psi_f,all_obs=False)[-1]
            
            #print(time.time()-ti)
            #print(current_fid) 
            #print(p_vals)
            
            old_p_vals=p_vals[:]
            p_vals=p_vals_new[:]
            
            new_fidelity=Fidelity(psi_i,H_fid,[t_vals[0],t_vals[-1]],basis=basis,psi_f=psi_f,all_obs=False)[-1]
            
            if new_fidelity > best_fid:
                best_fid=new_fidelity
                best_protocol=p_vals


            dF=-(new_fidelity-current_fid)
            if dF<0:                 
                donothing=0         #current_protocol=new_protocol
            elif np.random.uniform() < np.exp(-beta*dF):
                donothing=0         #current_protocol=new_protocol
            else:
                donothing=1
                p_vals=old_p_vals[:]
    
        T-=dT
     
    return best_fid,best_protocol
         
action_set=[-0.2,0.0,0.2]
 

best_fid,best_protocol=simulate_anneal(10,0.1,10,action_set)
print("Best protocol: ",best_protocol)
prtin("Fidelty:",best_fid)


#===============================================================================
# 
# 
# print(np.random.choice(action_set))
# exit()
# simulate_anneal(100,0.1,20,action_set)
#===============================================================================
 
 
 
 
