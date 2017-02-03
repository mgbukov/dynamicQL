#from Q_learning_c import Q_learning
from Q_learning import Q_learning
import tweezer_hamiltonian,descrete_SE,tweezer_hamiltonian
#from evaluate_data import load_data
import numpy as np

import time
import sys
import os
import gc


# model parameters
L=40
x = np.linspace(-1.0,1.0,L)
x0=-0.55 # initial tweezer position
d_tweezer=-160.0 # potential depth
#dA = in steps of 10

# prealocate kinetic energy
H_kin=0.5*descrete_SE.laplace_mat(x,x[1]-x[0])
# calculate initial and final state
psi_i,psi_f = tweezer_hamiltonian.target_states(H_kin,x,x0,d_tweezer)


print "overlap btw initial and target state is:", abs(psi_i.dot(psi_f))

max_t_steps = 120 #max_t_steps_vec[int(sys.argv[3])-1] #40 
delta_time = 1.0/200.0

##### RL params #####
var0_min, var0_max = [-1.0,1.0]

N_tilings = 100
N_tiles = 20

h_field = list( np.linspace(var0_min,var0_max,N_tiles) ) 
dh_field = h_field[1]-h_field[0]


########
# define RL  hyper params
state_i = np.array([x0])

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
physics_params = (L,max_t_steps,delta_time,x,x0,d_tweezer,psi_i,psi_f)


# initiate learning
Q_learning(*(RL_params+physics_params),save=True)




