#from Q_learning_c import Q_learning
from Q_learning import Q_learning
#from evaluate_data import load_data
from quspin.tools.measurements import ent_entropy
import Hamiltonian
import numpy as np

import time
import sys
import os
import gc

# make system update output files regularly
sys.stdout.flush()

### define save directory for data
# read in local directory path
str1=os.getcwd()
str2=str1.split('\\')
n=len(str2)
my_dir = str2[n-1]

max_t_steps_vec=np.array([4,8,10,12,20,30,40,48,50,60,70,80])   #np.linspace(5,50,10,dtype=int)


# define model params
L = int(sys.argv[4]) # system size
if L==1:
	J=0
else:
	J = 1.0 # zz interaction
hz = 1.0 # hz field
hx_i= -2.0 # initial hx coupling
hx_f= +2.0 # final hx coupling

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
	#E_i, psi_i = H.eigsh(time=0,k=1,sigma=-0.1,maxiter=1E10,return_eigenvectors=True)
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


print "number of states is:", H.Ns
print "initial and final energies are:", E_i, E_f
#print "initial entanglement is:", ent_entropy(psi_i,H.basis)['Sent']
print "overlap btw initial and target state is:", abs(psi_i.dot(psi_f)**2)

max_t_steps = max_t_steps_vec[int(sys.argv[3])-1] #40 
delta_time = 0.05 #0.05

##### RL params #####
var0_min, var0_max = [-4.0,4.0]

N_tilings = 100
N_tiles = 20

h_field = list( np.linspace(var0_min, var0_max,N_tiles) ) 
dh_field = h_field[1]-h_field[0]


########
# define RL  hyper params
state_i = np.array([-4.0])

# realisation number
N=int(sys.argv[2])-1
bang=int(sys.argv[1])-1 # 1: bang-bang; 0: continuous actions 
# number of episodes
N_episodes = 20001
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
m_expl=0.125 # 0.125

RL_params = (N,N_episodes,alpha_0,eta,lmbda,beta_RL_i,beta_RL_inf,T_expl,m_expl,N_tilings,N_tiles,state_i,h_field,dh_field,bang)
physics_params = (L,max_t_steps,delta_time,J,hz,hx_i,hx_f,psi_i,psi_f)



##### pre-calculate unitaries
if bang==0 and not os.path.isfile(my_dir+"/unitaries/unitaries_L={}_cont".format(L)+'.pkl'):
	Hamiltonian.Unitaries(delta_time,L,J,hz,0.1,var0_max,var0_min,state_i,save=True,save_str='_cont')
elif bang==1 and not os.path.isfile(my_dir+"/unitaries/unitaries_L={}_bang".format(L)+'.pkl'):
	Hamiltonian.Unitaries(delta_time,L,J,hz,8.0,var0_max,var0_min,state_i,save=True,save_str='_bang')

#exit()

# initiate learning
Q_learning(*(RL_params+physics_params),save=True)
#load_data(*(RL_params+physics_params))

