import numpy as np
import scipy.linalg as _la
import scipy.sparse.linalg as _sla
import numpy.random as random

import Reinforcement_Learning as RL
import Hamiltonian

from quspin.tools.measurements import ent_entropy
from quspin.operators import exp_op

import matplotlib.pyplot as plt
import pylab

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pylab

import time
import sys
import os

import cPickle

random.seed()

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


def Learn_Policy(state_i,theta,tilings,dims,actions,R):

	N_tilings, N_lintiles, N_vars = dims
	N_tiles = N_lintiles**N_vars
	shift_tile_inds = [j*N_tiles for j in xrange(N_tilings)]

	avail_actions = RL.all_actions()
	s = state_i.copy()

	for t_step, A in enumerate(actions):
		# calculate state-action indices
		indA = avail_actions.index(A)
		theta_inds = RL.find_feature_inds(s,tilings,shift_tile_inds)
			
		# check if max learnt
		if max(theta[theta_inds,t_step,:].ravel())>R/N_tilings and R>0:
			# Q function
			Q = np.sum(theta[theta_inds,t_step,:],axis=0)
			indA_max=np.argmax(Q)

			if max(Q) > 1E-13:
				theta[theta_inds,t_step,:]*=R/Q[indA_max]
			
		# calculate theta function
		theta[theta_inds,t_step,indA] = (R+1E-2)/(N_tilings)
		
		s+=A

	return theta



def Q_learning(RL_params,physics_params,theta=None,tilings=None):

	####################################################################
	start_time = time.time()
	####################################################################
	# display full strings
	np.set_printoptions(threshold='nan')
	######################################################################
	#######################   read off params 	 #########################
	######################################################################
	
	# read off RL_params
	RL_keys = ['N_episodes','alpha_0','eta','lmbda','beta_RL','traces','dims','N_tiles','state_i','h_field','dh_field']
	from numpy import array
	for key,value in RL_params.iteritems():
		#print key, repr(value)
		if key not in RL_keys:
			raise TypeError("Key '{}' not allowed for use in dictionary!".format(key))
		# turn key to variable and assign its value
		exec("{} = {}".format(key,repr(value)) ) in locals()

	# read off physics params
	physics_keys = ['L','max_t_steps','delta_t','J','hz','hx_i','hx_f','psi_i','psi_f','E_i','E_f']
	for key,value in physics_params.iteritems():
		#print key, repr(value)
		if key not in physics_keys:
			raise TypeError("Key '{}' not allowed for use in dictionary!".format(key))
		# turn key to variable and assign its value
		exec( "{} = {}".format(key,repr(value)) ) in locals()

	
	######################################################################
	# save data
	save = True
	N = 0 # realisation #

	# define all actions
	actions = RL.all_actions()

	# eta limits # max and min field
	hx_limits = [h_field[0],h_field[-1]]

	# get dimensions
	N_tilings, N_lintiles, N_vars = dims
	N_tiles = N_lintiles**N_vars
	N_actions = len(actions)
	shift_tile_inds = [j*N_tiles for j in xrange(N_tilings)]

	if theta is None:
		theta=np.zeros((N_tiles*N_tilings,max_t_steps,N_actions), dtype=np.float64)
	
	if tilings is None:
		tilings = RL.gen_tilings(h_field,dh_field,N_tilings)
   	
	# pre-allocate traces variable
	e = np.zeros_like(theta)
	fire_trace = np.ones(N_tilings)

	# pre-allocate usage vector: inverse gradient descent learning rate
	u0 = 1.0/alpha_0*np.ones((N_tiles*N_tilings,), dtype=np.float64)
	u=np.zeros_like(u0)	

	#### physical quantities

	# define ED Hamiltonian H(t)
	b=hx_i
	lin_fun = lambda t: b 
	# define Hamiltonian
	H = Hamiltonian.Hamiltonian(L,fun=lin_fun,**{'J':J,'hz':hz})
	# define matrix exponential; will be changed every time b is overwritten
	exp_H=exp_op(H,a=-1j*delta_t)
	
	# preallocate quantities
	Return_ave = np.zeros((N_episodes,),dtype=np.float64)
	Return = np.zeros_like(Return_ave)
	Fidelity_ep = np.zeros_like(Return_ave)

	# initialise best fidelity
	best_fidelity = 0.0 # best encountered fidelity
	
	# calculate importance sampling ratio
	R = 0.0 # instantaneous fidelity
	# preallocate physical state
	psi = np.zeros_like(psi_i)
		
	# loop over episodes
	for ep in xrange(N_episodes):
		# set traces to zero
		e *= 0.0
		# set initial usage vector
		u[:]=u0[:]

		# set initial state of episode
		S = state_i.copy()
		
		# get set of features present in S
		theta_inds = RL.find_feature_inds(S,tilings,shift_tile_inds)
		Q = np.sum(theta[theta_inds,0,:],axis=0) # for each action at time t_step=0

		# preallocate physical quantties
		psi[:] = psi_i[:] # quantum state at time
		
		# taken encountered and taken
		actions_taken = []
		
		# generate episode
		for t_step in xrange(max_t_steps): #

			# calculate available actions from state S
			avail_inds = np.argwhere((S[0]+np.array(actions) <= hx_limits[1])*(S[0]+np.array(actions) >= hx_limits[0])).squeeze()
			avail_actions = [actions[_j] for _j in avail_inds]

			# calculate greedy action(s) wrt Q policy
			A_greedy = avail_actions[ random.choice( np.argwhere(Q[avail_inds]==np.amax(Q[avail_inds])).ravel() ) ]
			
			# choose a random action
			P = np.exp(beta_RL*Q[avail_inds])
			p = np.cumsum(P/sum(P))
			A = avail_actions[np.searchsorted(p,random.uniform(0.0,1.0))]
			
			# find the index of A
			indA = actions.index(A)
					
			# reset traces if A is exploratory
			if abs(A - A_greedy) > np.finfo(A).eps:
				e *= 0.0

			# take action A, return state S_prime and actual reward R
			################################################################################
			######################    INTERACT WITH ENVIRONMENT    #########################
			################################################################################

			# define new state
			S_prime = S.copy()
			# calculate new field value 			
			S_prime[0] += A

			### assign reward
			R *= 0.0
			
			# all physics happens here
			b = S_prime[0]
			psi = exp_H.dot(psi)

			# assign reward
			if t_step == max_t_steps-1:
				# calculate final fidelity
				fidelity = abs( psi.conj().dot(psi_f) )**2
				# reward
				R += fidelity
				
			################################################################################
			################################################################################
			################################################################################
			
			# record action taken
			actions_taken.append(A)

			############################

			# calculate usage and alpha vectors: alpha_inf = eta
			u[theta_inds]*=(1.0-eta)
			u[theta_inds]+=1.0
			alpha = 1.0/(N_tilings*u[theta_inds])
			
			# Q learning update rule; GD error in time t
			delta = R - Q[indA] # error in gradient descent
			# TO
			Q_old = theta[theta_inds,t_step,indA].sum()
			
			# update traces
			e[theta_inds,t_step,indA] = alpha*fire_trace

			# check if S_prime is terminal or went out of grid
			if t_step == max_t_steps-1: 
				# update theta
				theta += delta*e
				# GD error in field h
				delta_TO = Q_old - theta[theta_inds,t_step,indA].sum()
				theta[theta_inds,t_step,indA] += alpha*delta_TO
				# go to next episode
				break

			# get set of features present in S_prime
			theta_inds_prime = RL.find_feature_inds(S_prime,tilings,shift_tile_inds)
			
			# t-dependent Watkin's Q learning
			Q = np.sum(theta[theta_inds_prime,t_step+1,:],axis=0)

			# update theta
			delta += max(Q)
			theta += delta*e

			# GD error in field h
			delta_TO = Q_old - theta[theta_inds,t_step,indA].sum()
			theta[theta_inds,t_step,indA] += alpha*delta_TO

			# update traces
			e[theta_inds,t_step,indA] -= alpha*e[theta_inds,t_step,indA].sum()
			e *= lmbda

			################################
			# S <- S_prime
			S[:] = S_prime[:]
			theta_inds[:]=theta_inds_prime[:]

		# record average return
		Return_ave[ep] = 1.0/(ep+1)*(R + ep*Return_ave[ep-1])
		Return[ep] = R
		Fidelity_ep[ep] = fidelity
	

		# if greedy policy completes a full episode and if greedy fidelity is worse than inst one
		if fidelity-best_fidelity>1E-12:
			# update list of best actions
			best_actions = actions_taken[:]
			# calculate best protocol and fidelity
			protocol_best,t_best = best_protocol(best_actions,hx_i,delta_t)
			R_best=R
			best_fidelity=fidelity
			# learn policy
			theta = Learn_Policy(state_i,theta,tilings,dims,best_actions,R_best)
			

		# force-learn best encountered every 100 episodes
		if (ep%40==0 and ep!=0) and (R_best is not None) and beta_RL<1E12:
			theta = Learn_Policy(state_i,theta,tilings,dims,best_actions,R_best)
				
		if ep%20 == 0:
			print "finished simulating episode {} with fidelity {} at hx_f = {}.".format(ep+1,np.round(fidelity,3),S_prime[0])
			print 'best encountered fidelity is {}.'.format(np.round(best_fidelity,3))
	
	# save data
	Data_fid = np.zeros((3,N_episodes))
	
	Data_fid[0,:] = Fidelity_ep
	Data_fid[1,:] = Return
	Data_fid[2,:] = Return_ave
	#
	Data_protocol = np.zeros((2,max_t_steps))

	Data_protocol[0,:] = t_best
	Data_protocol[1,:] = protocol_best

	if save:			
		args = (N,L) + tuple( np.around([J,hz,hx_i,hx_f] ,2) )
		# as txt format
		dataname  =  "RL_data_N=%s_L=%s_J=%s_hz=%s_hxi=%s_hxf=%s.txt"   %args
		np.savetxt(dataname,Data_fid.T)

		dataname  =  "protocol_data_N=%s_L=%s_J=%s_hz=%s_hxi=%s_hxf=%s.txt"   %args
		np.savetxt(dataname,Data_protocol.T)
		# save as pickle
		dataname  =  "theta_data_N=%s_L=%s_J=%s_hz=%s_hxi=%s_hxf=%s.pkl"   %args
		cPickle.dump(theta, open(dataname, "wb" ) )
		#cPickle.load(open(dataname, "rb" ))

		dataname  =  "tilings_data_N=%s_L=%s_J=%s_hz=%s_hxi=%s_hxf=%s.pkl"   %args
		cPickle.dump(tilings, open(dataname, "wb" ) )

		dataname  =  "RL_params_data_N=%s_L=%s_J=%s_hz=%s_hxi=%s_hxf=%s.pkl"   %args
		cPickle.dump(RL_params, open(dataname, "wb" ) )
		



		
	print "Calculating the Q function loop using Q-Learning took",("--- %s seconds ---" % (time.time() - start_time))


def best_protocol(best_actions,hx_i,delta_t):
	""" This function builds the best encounteres protocol from best_actions """
	protocol=[]
	t = [delta_t*_i for _i in range(len(best_actions))]
	s = hx_i
	for _i,a in enumerate(best_actions):
		s+=a
		protocol.append(s)
	return protocol, t
