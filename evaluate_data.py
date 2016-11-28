import numpy as np
import numpy.random as random

import Hamiltonian
import plot_data as plot
from quspin.operators import exp_op
from quspin.tools.measurements import ent_entropy

import time
import sys
import os
import cPickle

# only for printing data
def truncate(f,n):
    '''Truncates/pads a float f to n decimal places without rounding'''

    f_trunc=[]
    for f_j in f:
    	s = "{}".format(f_j)
    	if 'e' in s or 'E' in s:
    		f_trunc.append( '{0:.{1}f'.format(f,n))
    	else:
    		i,p,d = s.partition('.')
    		f_trunc.append( '.'.join([i,(d + '0'*n)[:n]]))
    return f_trunc

def load_data(N,N_episodes,alpha_0,eta,lmbda,beta_RL_i,beta_RL_inf,T_expl,m_expl,N_tilings,N_tiles,state_i,h_field,dh_field,
			   L,max_t_steps,delta_time,J,hz,hx_i,hx_f,psi_i,psi_f):
	
	args = (N,N_episodes,max_t_steps,L) + tuple( truncate([J,hz,hx_i,hx_f] ,2) )
	data_params = "_N=%s_Nep=%s_T=%s_L=%s_J=%s_hz=%s_hxi=%s_hxf=%s"   %args

	# as txt format
	dataname  =  "RL_data"+data_params+'.txt'
	Data_fid = np.loadtxt(dataname)

	dataname  =  "protocol_data"+data_params+'.txt'
	Data_protocol = np.loadtxt(dataname)
	# save as pickle
	dataname  =  "theta_data"+data_params+'.pkl'
	theta=cPickle.load(open(dataname, "rb" ))

	dataname  =  "tilings_data"+data_params+'.pkl'
	tilings=cPickle.load(open(dataname, "rb" ))

	
	Fidelity_ep=Data_fid[:,0]
	Return=Data_fid[:,1]
	Return_ave=Data_fid[:,2]

 	t_best=Data_protocol[:,0]
	protocol_best=Data_protocol[:,1] 
	protocol_greedy=Data_protocol[:,2] 


	# create plots
	plot.plot_rewards(Fidelity_ep,Return,Return_ave,'RL_stats',data_params)

	plot.observables(L,t_best,protocol_best,hx_i,hx_f,J,hz,data_params+'_best')
	plot.observables(L,t_best,protocol_greedy,hx_i,hx_f,J,hz,data_params+'_greedy')


