import numpy as np

#print("erhere")


import Hamiltonian
import numpy as np



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
lin_fun = lambda t: m*t + b
# define Hamiltonian
H, basis = Hamiltonian.Hamiltonian(L,fun=lin_fun,**{'J':J,'hz':hz})
# defien Hamiltonian for interpolated fidelity
t_vals, p_vals = [0.0,0.0], [0.0,0.0]
protocol_fun = lambda t: np.interp(t, t_vals, p_vals)
H, basis = Hamiltonian.Hamiltonian(L,fun=lin_fun,**{'J':J,'hz':hz})


# calculate initial state
E_i, psi_i = H.eigsh(time=0,k=1,sigma=-100.0)
E_i = E_i.squeeze()
psi_i = psi_i.squeeze()
# calculate final state
b = hx_f
E_f, psi_f = H.eigsh(time=0,k=1,sigma=-100.0)
E_f = E_f.squeeze()
psi_f = psi_f.squeeze()

print(Fidelity(psi_i,H,[t_vals[0],t_vals[-1]],basis=basis,psi_f=psi_f,all_obs=False)[-1])


#===============================================================================
#===============================================================================
# action_set=np.array([-0.2,0.1,-0.5,10.])
# cp=action_set[np.random.choice([0,1,2,3],20)]
# print(cp)
# print(np.array([np.sum(cp[:t]) for t in range(20)]))r
# exit()
#===============================================================================
#===============================================================================

 
def simulate_anneal(Ti,dT,episode_time,action_set):
    T=Ti
    N=1000
       
    current_protocol=action_set[np.random.choice(action_set,episode_time)]
    current_h_protocol=np.array([np.sum(current_protocol[:t]) for t in range(episode_time)])
    
    #current_state=
       
    while T>0.:
        beta=1./T
        for i in range(1000):
            rand_pos=np.random.randint(L)
            new_protocol=current_protocol
            new_protocol[rand_pos]=action_set[np.random_choice(action_set,1)]
            dF=-(Fidelity(new_protocol)-Fidelity(current_protocol))
            
            if dF<0:
                current_protocol=new_protocol
            elif np.random.uniform() < np.exp(-beta*dF):
                current_protocol=new_protocol
        T-=dT
    
    return current_protocol
        






















#Fidelity(psi_i,H_fid,t_vals,basis=None,psi_f=None,all_obs=False)



#psi_t = H_fid.evolve(psi_i,t_vals[0],t_vals,iterate=True,atol=1E-12,rtol=1E-12)