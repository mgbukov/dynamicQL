import numpy as np
import numpy.random as random

import Hamiltonian
from quspin.operators import exp_op

import time
import sys
import os
import cPickle

random.seed(0)

def best_protocol(best_actions,hx_i,delta_t):
	""" This function builds the best encounteres protocol from best_actions """
	protocol=np.zeros_like(best_actions)
	t = np.array([delta_t*_i for _i in range(len(best_actions))])
	S = hx_i
	for _i,A in enumerate(best_actions):
		S+=A
		protocol[_i]=S
	return protocol, t


def Learn_Policy(state_i,best_actions,R,theta,tilings,actions):

	N_tilings = tilings.shape[0]
	N_tiles = tilings.shape[1]
	
	# preallocate theta_inds
	theta_inds_zeros=np.zeros((N_tilings,),dtype=int)

	S = state_i.copy()

	for t_step, A in enumerate(best_actions):
		# calculate state-action indices
		indA = np.searchsorted(actions,A)
		
		theta_inds=theta_inds_zeros
		for _k, tiling in enumerate(tilings): # cython-ise this loop as inline fn!
			idx = tiling.searchsorted(S)
			idx = np.clip(idx, 1, len(tiling)-1)
			left, right = tiling[idx-1], tiling[idx]
			idx -= S - left < right - S
			theta_inds[_k]=idx[0]+_k*N_tiles
			
		# check if max learnt
		if max(theta[theta_inds,t_step,:].ravel())>R/N_tilings and R>0:
			Q = np.sum(theta[theta_inds,t_step,:],axis=0)
			indA_max=np.argmax(Q)

			if max(Q) > 1E-13:
				theta[theta_inds,t_step,:]*=R/Q[indA_max]
			
		# calculate theta function
		theta[theta_inds,t_step,indA] = (R+1E-2)/(N_tilings)
		
		S+=A

	return theta



def Q_learning(N,N_episodes,alpha_0,eta,lmbda,beta_RL,N_tilings,N_tiles,state_i,h_field,dh_field,
			   L,max_t_steps,delta_t,J,hz,hx_i,hx_f,psi_i,psi_f,
			   theta=None,tilings=None,save=False):
	"""
	This function applies modified Watkins' Q-Learning for time-dependent states with
	force-learn replays.

	1st row: RL arguments
	2nd row: physics arguments
	3rd row: optional arguments
	"""
	######################################################################

	##### physical quantities ######

	# define ED Hamiltonian H(t)
	b=hx_i
	lin_fun = lambda t: b 
	# define Hamiltonian
	H = Hamiltonian.Hamiltonian(L,fun=lin_fun,**{'J':J,'hz':hz})
	# define matrix exponential; will be changed every time b is overwritten
	exp_H=exp_op(H,a=-1j*delta_t)
	# preallocate physical state
	psi = np.zeros_like(psi_i)

	##### RL quantities	#####

	# define actions
	pos_actions=[0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0]
	#pos_actions=[2.0]
	neg_actions=[-i for i in pos_actions]
	actions = np.sort(neg_actions + [0.0] + pos_actions)
	del pos_actions,neg_actions

	N_actions = len(actions)
	
	if theta is None:
		theta=np.zeros((N_tiles*N_tilings,max_t_steps,N_actions), dtype=np.float64)
	
	if tilings is None:
		tilings = np.array([h_field + np.random.uniform(0.0,dh_field,1) for j in xrange(N_tilings)])
		
	# pre-allocate traces variable
	e = np.zeros_like(theta)
	fire_trace = np.ones(N_tilings)

	# pre-allocate usage vector: inverse gradient descent learning rate
	u0 = 1.0/alpha_0*np.ones((N_tiles*N_tilings,), dtype=np.float64)
	u=np.zeros_like(u0)	
	
	# preallocate quantities
	Return_ave = np.zeros((N_episodes,),dtype=np.float64)
	Return = np.zeros_like(Return_ave)
	Fidelity_ep = np.zeros_like(Return_ave)

	# initialise best fidelity
	best_fidelity = 0.0 # best encountered fidelity
	# initialise reward
	R = 0.0 
	# preallocate theta_inds
	theta_inds_zeros=np.zeros((N_tilings,),dtype=int)
		
	# loop over episodes
	for ep in xrange(N_episodes):
		# set traces to zero
		e *= 0.0
		# set initial usage vector
		u[:]=u0[:]

		# set initial state of episode
		S = state_i.copy()
		
		# get set of features present in S
		theta_inds=theta_inds_zeros
		for _k, tiling in enumerate(tilings): # cython-ise this loop as inline fn!
			idx = tiling.searchsorted(S)
			idx = np.clip(idx, 1, len(tiling)-1)
			left, right = tiling[idx-1], tiling[idx]
			idx -= S - left < right - S
			theta_inds[_k]=idx[0]+_k*N_tiles
		Q = np.sum(theta[theta_inds,0,:],axis=0) # for each action at time t_step=0

		# preallocate physical quantties
		psi[:] = psi_i[:] # quantum state at time
		
		# taken encountered and taken
		actions_taken = np.zeros((max_t_steps,),dtype=np.float64)
		
		# generate episode
		for t_step in xrange(max_t_steps): 

			# calculate available actions from state S
			avail_inds = np.argwhere((S[0]+np.array(actions)<=h_field[-1])*(S[0]+np.array(actions)>=h_field[0])).squeeze()
			avail_actions = actions[avail_inds]

			# calculate greedy action(s) wrt Q policy
			A_greedy = avail_actions[random.choice(np.argwhere(Q[avail_inds]==np.amax(Q[avail_inds])).ravel() ) ]
			
			# choose a random action
			P = np.exp(beta_RL*Q[avail_inds])
			A = avail_actions[np.searchsorted(np.cumsum(P/np.sum(P)),random.uniform(0.0,1.0))]
			
			# find the index of A
			indA = np.searchsorted(actions,A)
					
			# reset traces if A is exploratory
			if abs(A - A_greedy) > np.finfo(A).eps:
				e *= 0.0

			# record action taken
			actions_taken[t_step]=A

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
				fidelity = abs(psi.conj().dot(psi_f))**2
				# reward
				R += fidelity
				
			################################################################################
			################################################################################
			################################################################################

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
			theta_inds_prime = theta_inds_zeros
			for _k, tiling in enumerate(tilings): # cython-ise this loop as inline fn!
				idx = tiling.searchsorted(S_prime)
				idx = np.clip(idx, 1, len(tiling)-1)
				left, right = tiling[idx-1], tiling[idx]
				idx -= S_prime - left < right - S_prime
				theta_inds_prime[_k]=idx[0]+_k*N_tiles
			
			# t-dependent Watkin's Q learning
			Q = np.sum(theta[theta_inds_prime,t_step+1,:],axis=0)

			# update theta
			delta += np.max(Q)
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
			# best reward and fidelity
			R_best=R
			best_fidelity=fidelity
			# learn policy
			theta = Learn_Policy(state_i,best_actions,R_best,theta,tilings,actions)
			

		# force-learn best encountered every 100 episodes
		if (ep%40==0 and ep!=0) and (R_best is not None) and beta_RL<1E12:
			theta = Learn_Policy(state_i,best_actions,R_best,theta,tilings,actions)
				
		if ep%20 == 0:
			print "finished simulating episode {} with fidelity {} at hx_f = {}.".format(ep+1,np.round(fidelity,3),S_prime[0])
			print 'best encountered fidelity is {}.'.format(np.round(best_fidelity,3))
	

	# calculate best protocol and fidelity
	protocol_best,t_best = best_protocol(best_actions,hx_i,delta_t)

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

