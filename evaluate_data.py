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

def load_data(N,N_episodes,alpha_0,eta,lmbda,beta_RL_i,beta_RL_inf,T_expl,m_expl,N_tilings,N_tiles,state_i,h_field,dh_field,bang,
			   L,max_t_steps,delta_time,J,hz,hx_i,hx_f,psi_i,psi_f):

	if bang:
		a_str='_bang'
	else:
		a_str='_cont'
	
	args = (N,N_episodes,max_t_steps,L) + tuple( truncate([J,hz,hx_i,hx_f] ,2) )
	data_params = "_N=%s_Nep=%s_T=%s_L=%s_J=%s_hz=%s_hxi=%s_hxf=%s"   %args
	data_params+=a_str

	save_dir = "data/"

	# as txt format
	dataname  =  save_dir + "RL_data"+data_params+'.txt'
	Data_fid = np.loadtxt(dataname)

	dataname  =  save_dir + "protocol_data"+data_params+'.txt'
	Data_protocol = np.loadtxt(dataname)
	# save as pickle
	dataname  =  save_dir + "theta_data"+data_params+'.pkl'
	theta=cPickle.load(open(dataname, "rb" ))

	dataname  =  save_dir + "tilings_data"+data_params+'.pkl'
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



def real_ave(N,N_episodes,alpha_0,eta,lmbda,beta_RL_i,beta_RL_inf,T_expl,m_expl,N_tilings,N_tiles,state_i,h_field,dh_field,bang,
			 L,max_t_steps,delta_time,J,hz,hx_i,hx_f,psi_i,psi_f,
			 log_scale=False):

	max_t_steps_vec=np.linspace(5,50,10,dtype=int)
	Nmax=100

	save_dir = "../data_RL/"
	if bang:
		a_str='bang'
	else:
		a_str='cont'

	# pre-allocate data
	Data_RL = np.zeros((N_episodes,3,len(max_t_steps_vec),Nmax))
	Best_fid = np.zeros((N_episodes,Nmax))




	fid_vs_T = []
	fid_vs_T_ave = []

	

	for j,max_t_steps in enumerate(max_t_steps_vec):

		# pre-allocate data
		Fidelity = np.zeros((max_t_steps+1,Nmax))
		E = np.zeros_like(Fidelity)
		delta_E=np.zeros_like(Fidelity)
		Sd=np.zeros_like(Fidelity)

		Data_protocol = np.zeros((max_t_steps,3,Nmax))

		for N in range(Nmax):

			args = (N,N_episodes,max_t_steps,L) + tuple( truncate([J,hz,hx_i,hx_f] ,2) )
			data_params = "_N=%s_Nep=%s_T=%s_L=%s_J=%s_hz=%s_hxi=%s_hxf=%s"   %args
			data_params+="_"+a_str

			dataname  =  save_dir + "RL_data"+data_params+'.txt'
			Data_RL[:,:,j,N] = np.loadtxt(dataname)
			Best_fid[:,N] = Data_RL[:,0,j,N].squeeze()

			dataname  =  save_dir + "protocol_data"+data_params+'.txt'
			Data_protocol[:,:,N] = np.loadtxt(dataname)

			t_best = Data_protocol[:,0,N]
			protocol_best = Data_protocol[:,1,N]

			aux1,aux2,aux3,aux4 = plot.observables(L,t_best,protocol_best,hx_i,hx_f,J,hz,data_params,fore_str='best_ave_',plot_data=False)

			Fidelity[:,N]=aux1
			E[:,N]=aux2
			delta_E[:,N]=aux3
			Sd[:,N]=aux4

		
			print (j,N)
			 
		
		# average over realisations
		ave_dataRL = np.mean(Data_RL,axis=3).squeeze()
		ave_F = np.mean(Fidelity,axis=1).squeeze()
		ave_E = np.mean(E,axis=1).squeeze()
		ave_dE = np.mean(delta_E,axis=1).squeeze()
		ave_Sd = np.mean(Sd,axis=1).squeeze()


		# keep only best encountered fidelity
		best_fid=np.zeros((N_episodes,))
		aux=np.mean(Best_fid,axis=1).squeeze()
		for _i,_j in enumerate(aux):

			if _i==0:
				best_fid[_i]=_j
			elif np.all(_j > aux[0:_i]):
				best_fid[_i]=_j
			else:
				best_fid[_i]=best_fid[_i-1]

		# best fidelity
		print Best_fid.max()
		fid_vs_T.append(Best_fid.max())
		j_max = np.where(Best_fid==fid_vs_T[j])[1][0]
		fid_vs_T_ave.append(best_fid[-1])

		# plot resuts
		args = (N_episodes,max_t_steps,L) + tuple( truncate([J,hz,hx_i,hx_f] ,2) )
		data_params = "_Nep=%s_T=%s_L=%s_J=%s_hz=%s_hxi=%s_hxf=%s"   %args
		

		### plot results
		# RL data
		plot.plot_rewards(ave_dataRL[:,0,j].squeeze(),ave_dataRL[:,1,j].squeeze(),ave_dataRL[:,2,j].squeeze(),a_str+'_ave_RL_stats',data_params,log_scale=False)
		# RL best reward in log scale
		plot.plot_rewards(best_fid,[],[],a_str+'_best_ave_fid',data_params,log_scale=True,fid_only=True)
		# observables
		plot.plot_protocols(t_best,None,ave_F,a_str+'_best_ave_fid',data_params,save=True)
		plot.plot_protocols(t_best,None,ave_E,a_str+'_best_ave_en',data_params,save=True)
		plot.plot_protocols(t_best,None,ave_dE,a_str+'_best_ave_en_fluct',data_params,save=True)
		plot.plot_protocols(t_best,None,ave_Sd,a_str+'_best_ave_sd',data_params,save=True)

		# best observables of all Nmax at a given T
		plot.observables(L,t_best,Data_protocol[:,1,j_max],hx_i,hx_f,J,hz,data_params,fore_str=a_str+'_best_real_')

	# plot best fidelity vs total ramp time and its realisation average
	args = (N_episodes,L) + tuple( truncate([J,hz,hx_i,hx_f] ,2) )
	data_params = "_Nep=%s_L=%s_J=%s_hz=%s_hxi=%s_hxf=%s"   %args
	
	plot.plot_best_fid_vs_T(delta_time*max_t_steps_vec,fid_vs_T,fid_vs_T_ave,a_str+'_best_fid_vs_T',data_params)

