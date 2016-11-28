from Q_learning import Q_learning
import Hamiltonian
import numpy as np


# define model params
L = 1 # system size
if L==1:
	J = 0.0 # required by PBC
	hz = 1.0
	hx_i= -1.0 # initial hx coupling
	hx_f= +1.0 # final hx coupling
else:
	J = -1.0 #/0.809 # zz interaction
	hz = 0.5 #0.9045/0.809 #1.0 # hz field
	hx_i= 0.0 # initial hx coupling
	hx_f= 2.0 # final hx coupling

# define dynamic params of H(t)
b=hx_i
lin_fun = lambda t: b
# define Hamiltonian
H_params = {'J':J,'hz':hz}
H = Hamiltonian.Hamiltonian(L,fun=lin_fun,**H_params)

# calculate initial state
if L==1:
	E_i, psi_i = H.eigh()
else:
	E_i, psi_i = H.eigsh(time=0,k=2,which='BE',maxiter=1E10,return_eigenvectors=True)
E_i = E_i[0]
psi_i = psi_i[:,0]
# calculate final state
b = hx_f
if L==1:
	E_f, psi_f = H.eigh()
else:
	E_f, psi_f = H.eigsh(time=0,k=2,which='BE',maxiter=1E10,return_eigenvectors=True)
E_f = E_f[0]
psi_f = psi_f[:,0]

max_t_steps = 20 #40 
delta_time = 0.05 #0.05

print "number of states is:", H.Ns
print "initial and final energies are:", E_i, E_f


##### RL params #####
var0_min, var0_max = [-4.0,4.0]

N_tilings = 100
N_tiles = 20

h_field = list( np.linspace(var0_min, var0_max,N_tiles) ) 
dh_field = h_field[1]-h_field[0]


########
# define RL  hyper params
hx_i = -4.0
state_i = np.array([hx_i])

# realisation number
N=0
# number of episodes
N_episodes = 2001
# learning rate
alpha_0 = 0.9
# usage or "u" eta: alpha_0 decays to eta in long run
eta = 0.6
# TD(lambda) parameter
lmbda = 1.0
# softmax exploration inverse temperature
beta_RL_i = 2.0
beta_RL_inf = 100.0
# set exploration period duration
T_expl=20
m_expl=0.25 # 0.125

RL_params = (N,N_episodes,alpha_0,eta,lmbda,beta_RL_i,beta_RL_inf,T_expl,m_expl,N_tilings,N_tiles,state_i,h_field,dh_field)
physics_params = (L,max_t_steps,delta_time,J,hz,hx_i,hx_f,psi_i,psi_f)


# initiate learning
Q_learning(*(RL_params+physics_params),save=True)

