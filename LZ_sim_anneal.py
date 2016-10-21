import numpy as np

#print("erhere")


import Hamiltonian
import numpy as np
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

# define model params
L = 10 # system size
J = 1.0/0.809 # zz interaction
hz = 0.2 #0.9045/0.809 #1.0 # hz field

hx_i = 0.001#-1.0 # initial hx coupling
hx_f = 2.0 #+1.0 # final hx coupling

# define ED Hamiltonian H(t)
m=0.0
b=hx_i
#lin_fun = lambda t: m*t + b

t_vals, p_vals = [0.0,1.0,2.0], [0.0,0.2,0.4]
protocol_fun = lambda t: np.interp(t, t_vals, p_vals)

#def lin_fun(t):
    #print p_vals
#    return np.interp(t, t_vals, p_vals)
# define Hamiltonian

H, basis = Hamiltonian.Hamiltonian(L,fun=protocol_fun,**{'J':J,'hz':hz})
# defien Hamiltonian for interpolated fidelity
#global t_vals,p_vals
#protocol_fun = lambda t: np.interp(t, t_vals, p_vals)

H_fid,_ = Hamiltonian.Hamiltonian(L,fun=protocol_fun,**{'J':J,'hz':hz})




# calculate initial state
E_i, psi_i = H.eigsh(time=0,k=1,sigma=-100.0)
E_i = E_i.squeeze()
psi_i = psi_i.squeeze()

b = hx_f
E_f, psi_f = H.eigsh(time=0,k=1,sigma=-100.0)
E_f = E_f.squeeze()
psi_f = psi_f.squeeze()

print(Fidelity(psi_i,H_fid,[t_vals[0],t_vals[-1]],basis=basis,psi_f=psi_f,all_obs=False)[-1])

p_vals=[0.0,0.2,0.8]

print(Fidelity(psi_i,H_fid,[t_vals[0],t_vals[-1]],basis=basis,psi_f=psi_f,all_obs=False)[-1])

print(psi_i.shape)
print(abs(psi_i.conj().dot(psi_f))**2)
exit()


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


action_set=[-0.2,0.,0.2]
hi=-1.
episode_length=6

ap,traj=random_trajectory([-0.2,0.,0.2],-1.,6)
print("Initial action protocol",ap)
print("Initial trajectory",traj)

ap_new,traj_new=propose_new_trajectory(ap,action_set,hi,episode_length)

print("New action protocol",ap_new)
print("New trajectory",traj_new)


def simulate_anneal(Ti,dT,episode_length,action_set):
    T=Ti
    N=1000
    hi=-1.
     
    action_protocol,p_vals=random_trajectory(action_set,hi,episode_length)
    t_vals=range(episode_length+1)
    print(action_protocol)
    print(p_vals)
    print(t_vals)
    #exit()
     
    #current_state=
     
    while T>0.:
        beta=1./T
        for i in range(1000):
            
            action_protocol_new,p_vals_new=propose_new_trajectory(action_protocol,action_set,hi,episode_length)
            
            current_fid=Fidelity(psi_i,H_fid,[t_vals[0],t_vals[-1]],basis=basis,psi_f=psi_f,all_obs=False)[-1]
            print(current_fid) 
            print(p_vals)
            
            old_p_vals=p_vals[:]
            p_vals=p_vals_new[:]
            
            new_fidelity=Fidelity(psi_i,H_fid,[t_vals[0],t_vals[-1]],basis=basis,psi_f=psi_f,all_obs=False)[-1]
            print(p_vals)
            print(new_fidelity)
            dF=-(new_fidelity-current_fid)
             
            print(dF)
            exit()
            if dF<0:
                current_protocol=new_protocol
            elif np.random.uniform() < np.exp(-beta*dF):
                current_protocol=new_protocol
        T-=dT
     
    return current_protocol
         
action_set=[-0.2,0.0,0.2]
 
simulate_anneal(100,0.01,10,action_set)


print(np.random.choice(action_set))
exit()
simulate_anneal(100,0.1,20,action_set)
 
 
 
 
