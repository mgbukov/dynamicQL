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

random.seed(0)

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


def Fidelity(psi_i,H_fid,t_vals,delta_t,psi_f=None,all_obs=False,Vf=None):
	''' This function calculates the physical quantities given a time-dep Hamiltonian H_fid.
		If psi_f is not given, then it returns the fidelity in the instantaneous eigenstate, otherwise
		--- the fidelity in the final state. '''
	basis = H_fid.basis
	
	# evolve state
	#psi_t = H_fid.evolve(psi_i,t_vals[0],t_vals,iterate=True,atol=1E-12,rtol=1E-12)


	# fidelity
	fid = []
	if all_obs:
		# get sysstem size
		L = basis.L
		# entanglement entropy density
		Sent=[]
		subsys = [j for j in range(L/2)]
		# energy
		E=[]
		# energy fluctuations
		dE=[]
		Sd=[]

	psi=psi_i.copy()
	for i, t_i in enumerate(t_vals):
	#for i, psi in enumerate(psi_t):

		psi = exp_op(H_fid(time=t_i),a=-1j*delta_t).dot(psi)

		if psi_f is not None:
			# calculate w.r.t. final state
			psi_target = psi_f
		else:
			# calculate w.r.t. instantaneous state
			_,psi_inst = H_fid.eigsh(time=t_vals[i],k=1,sigma=-100.0)
			psi_target = psi_inst.squeeze()

		fid.append( abs(psi.conj().dot(psi_target))**2 )

		#print i, abs(psi.conj().dot(psi_target))**2 

		if all_obs:

			if L>1:
				EGS,_ = H_fid.eigsh(time=t_vals[i],k=2,which='BE',maxiter=1E10,return_eigenvectors=False)
			else:
				EGS = H_fid.eigvalsh(time=t_vals[i])[0]

			E.append( H_fid.matrix_ele(psi,psi,time=t_vals[i]).real/L - EGS/L)
			if i==0:
				dE.append(0.0)
			else:
				dE.append( np.sqrt( (H_fid*H_fid(time=t_vals[i])).matrix_ele(psi,psi,time=t_vals[i]).real/L**2- E[i]**2  )   )
			#print  H_fid2.matrix_ele(psi,psi,time=t_vals[i]).real/L**2 - E[i]**2

			pn = abs( Vf.conj().T.dot(psi) )**2.0 + np.finfo(psi[0].dtype).eps
			Sd.append( -pn.dot(np.log(pn))/L )

			if basis.L!=1:
				Sent.append( ent_entropy(psi,basis,chain_subsys=subsys)['Sent'] )
			else:
				Sent.append(float('nan'))


	if not all_obs:
		return fid
	else:
		return fid , E, dE, Sent, Sd


def Replay(N_replay,RL_params,physics_params,theta,tilings,actions,R):
	""" This function replays a protocol in a greedy fashion. 
	After many iterations, this becomes equivalent to the function Learn_Policy. 
	"""

	####################################################################
	# display full strings
	np.set_printoptions(threshold='nan')
	######################################################################
	#######################   read off params 	 #########################
	######################################################################
	
	# read off RL_params
	RL_keys = ['N_episodes','gamma','alpha_0','eta','lmbda','eps','traces','dims','N_tiles','state_i','Vars','dVars','mu']
	from numpy import array
	for key,value in RL_params.iteritems():
		#print key, repr(value)
		if key not in RL_keys:
			raise TypeError("Key '{}' not allowed for use in dictionary!".format(key))
		# turn key to variable and assign its value
		exec("{} = {}".format(key,repr(value)) ) in locals()

	# read off physics params
	physics_keys = ['L','max_t_steps','delta_t','J','hz','hx_i',
				    'hx_f','psi_i','psi_f','E_i','E_f']
	for key,value in physics_params.iteritems():
		#print key, repr(value)
		if key not in physics_keys:
			raise TypeError("Key '{}' not allowed for use in dictionary!".format(key))
		# turn key to variable and assign its value
		exec( "{} = {}".format(key,repr(value)) ) in locals()

	avail_actions = RL.all_actions()

	N_tilings, N_lintiles, N_vars = dims
	N_tiles = N_lintiles**N_vars
	N_actions = len(avail_actions)
	shift_tile_inds = [j*N_tiles for j in xrange(N_tilings)]

	lmbda=1.0

	u0 = 1.0/alpha_0*np.ones((N_tiles*N_tilings,), dtype=np.float64)

	e = np.zeros(theta.shape, dtype=np.float64)
	# set trace function
	trace_fn = eval('RL.E_traces_'+traces)

	for j in xrange(N_replay):

		S = state_i.copy()

		# set traces to zero
		e *= 0.0
		# set initial usage vector
		u = u0.copy()

		theta_inds = RL.find_feature_inds(S,tilings,shift_tile_inds)
		Q = np.sum(theta[theta_inds,0,:],axis=0)
		E=0.0
		Q_old=[0.0 for i in avail_actions]

		
		for t_step in xrange(len(actions)):

			A = actions[t_step]
			indA = avail_actions.index(A)

			S[0]+=A

			u[theta_inds] *= (1.0-eta)
			u[theta_inds] += 1.0 

			with np.errstate(divide='ignore'):
				alpha = 1.0/(N_tilings*u[theta_inds])
				alpha[u[theta_inds]<1E-12] = 1.0

			# Q learning update rule 
			delta = 0.0 - Q[indA]

			if t_step > 0:
				delta_TO = Q_old[indA] - np.sum(theta[theta_inds,t_step-1,indA],axis=0)
			else:
				delta_TO=0.0

			e[theta_inds,t_step,indA] = alpha*(trace_fn(e[theta_inds,t_step,indA],alpha) - lmbda*E) 
	
			# theta <-- theta + \alpha*[theta(t-1)\phi(S) - theta(t)\phi(S)]
			theta[theta_inds,t_step,indA] += alpha*delta_TO
			
			# check if S_prime is terminal or went out of grid
			if t_step == max_t_steps-1: 
				# update theta
				theta += (R+delta)*e
				break


			theta_inds = RL.find_feature_inds(S,tilings,shift_tile_inds)

			# TO
			Q_old = np.sum(theta[theta_inds,t_step,:],axis=0)
			# non-TO
			Q = np.sum(theta[theta_inds,t_step+1,:],axis=0)

			# update theta and e
			indA_prime = avail_actions.index(actions[t_step+1])
			
			delta += Q[indA_prime]
			theta += delta*e
			
			#user_input = raw_input("save data? (y or n) ")
			#print t_step, np.round( np.sum(  theta[ RL.find_feature_inds(S,tilings,N_tiles),t_step,: ],axis=0), 3)
		

			E = np.sum(e[theta_inds,t_step,indA],axis=0)
			e *= lmbda

		"""
		print '____REPLAY_____', j
		s = hx_i
		t_step=0
		print "reward is", R 
		for a in actions:
			theta_inds = RL.find_feature_inds(s,tilings,shift_tile_inds)
			'''
			print 'state', s
			print 'traces'
			print np.sum( e[theta_inds,t_step,:], axis=0)
			print '++++++'
			print 'thetas'
			'''
			print np.round( np.sum(theta[theta_inds,t_step,:],axis=0), 3), np.around(s,3)
			t_step+=1
			s+=a
		print '__end__REPLAY_____'
		
		"""
	#user_input = raw_input("continue? (y or n) ")	
	return theta


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
			

		#'''
		# check if max learnt
		if max(theta[theta_inds,t_step,:].ravel())>R/N_tilings and R>0:
			# Q function
			Q = np.sum(theta[theta_inds,t_step,:],axis=0)
			indA_max=np.argmax(Q)
			# rescale theta
			#print '1.', theta[theta_inds,t_step,indA_max].sum(), R
			if max(Q) > 1E-13:
				theta[theta_inds,t_step,:]*=R/Q[indA_max]
			#print R/Q[indA_max], R, Q[indA_max]
			#print '2.', theta[theta_inds,t_step,indA_max].sum(), R
			#print abs(max(np.sum(theta[theta_inds,t_step,:],axis=0)) - R )>1E-12
			#print max(theta[theta_inds,t_step,:].ravel()), R/N_tilings
			

		# calculate theta function
		theta[theta_inds,t_step,indA] = (R+1E-2)/(N_tilings)
		
		"""
		if abs(max(np.sum(theta[theta_inds,t_step,:],axis=0)) - (R+1E-2) )>1E-12  and R>0:
			print 'BUGG'
			exit()
		"""
		s+=A

	return theta



def Q_learning(RL_params,physics_params,theta=None,tilings=None,greedy=False):

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

	if not greedy:
		# pre-allocate usage vector: inverse gradient descent learning rate
		u0 = 1.0/alpha_0*np.ones((N_tiles*N_tilings,), dtype=np.float64)
	else:
		u0 = np.inf*np.ones((N_tiles*N_tilings,), dtype=np.float64)
	u=np.zeros_like(u0)

	# set terminate episode variable for wandering off the h grid
	terminate = False
	

	#### physical quantities

	# define ED Hamiltonian H(t)
	b=hx_i
	lin_fun = lambda t: b #+ m*t
	# define Hamiltonian
	H = Hamiltonian.Hamiltonian(L,fun=lin_fun,**{'J':J,'hz':hz})
	# define matrix exponential
	exp_H=exp_op(H,a=-1j*delta_t)
	#"""
	''' will not need onless we plot '''
	# defien Hamiltonian for any step-lie protocol p_vals at times t_vals
	t_vals, p_vals = [0.0,0.0], [0.0,0.0]
	def step_protocol(t):
		return p_vals[np.argmin( abs(np.asarray(t_vals)-t) )]
	H_fid = Hamiltonian.Hamiltonian(L,fun=step_protocol,**{'J':J,'hz':hz})
	# calculate final basis
	b=hx_f
	_,Vf = H.eigh(time=0.0)
	b=hx_i
	''' will not need '''
	#"""
	
	# average reward
	Return_ave = np.zeros((N_episodes,1),dtype=np.float64)
	Return = Return_ave.copy()
	FidelitY = Return_ave.copy()

	# initialise best fidelity
	best_fidelity = 0.0 # best encountered fidelity
	# set of actions for best encountered protocol
	best_actions=[random.choice(actions) for j in range(max_t_steps)] 

	# calculate importance sampling ratio
	R = 0.0 # instantaneous fidelity
	R_best=None # fidelity of best encountered protocol

	psi = np.zeros_like(psi_i)
		
	# loop over episodes
	for j in xrange(N_episodes):
		# set traces to zero
		e *= 0.0
		# set initial usage vector
		u[:]=u0[:]

		# set initial state of episode
		S = state_i.copy()
		
		# get set of features present in S
		theta_inds = RL.find_feature_inds(S,tilings,shift_tile_inds)
		Q = np.sum(theta[theta_inds,0,:],axis=0) # for each action at time t_step=0

		E=0.0 # auxiliary for traces e
		Q_old=[0.0 for i in actions]

		# preallocate physical quantties
		psi[:] = psi_i[:] # quantum state at time

		protocol_inst = []
		t_inst = []
		
		# calculate fidelity for each fixed episode
		Return_j = 0.0

		# taken encountered and taken
		actions_taken = []

		
		# generate episode
		for t_step in xrange(max_t_steps): #

			# calculate available actions from state S
			avail_inds = np.argwhere((S[0]+np.array(actions) <= hx_limits[1])*(S[0]+np.array(actions) >= hx_limits[0])).squeeze()
			avail_actions = [actions[_j] for _j in avail_inds]

			# calculate greedy action(s) wrt Q policy
			#A_greedy = actions[random.choice( np.argwhere(Q==np.amax(Q)).ravel() )]
			A_greedy = avail_actions[ random.choice( np.argwhere(Q[avail_inds]==np.amax(Q[avail_inds])).ravel() ) ]
			
			# choose a random action
			P = np.exp(beta_RL*Q[avail_inds])
			p = np.cumsum(P/sum(P))
			if greedy or beta_RL>1E12:
				A = A_greedy
			else:
				A = avail_actions[np.searchsorted(p,random.uniform(0.0,1.0))]
			
			# find the index of A
			indA = actions.index(A)
					
			# reset traces and calculate probability under beh policy
			if abs(A - A_greedy) > np.finfo(A).eps:
				e *= 0.0

			# take action A, return state S_prime and actual reward R
			################################################################################
			######################    INTERACT WITH ENVIRONMENT    #########################
			################################################################################

			######################## 
			####  Landau_Zener  ####
			########################

			# define new state
			S_prime = S.copy()
			# calculate new field value 			
			S_prime[0] += A

			### assign rewards
			R *= 0.0

			###########################
			######  Environment  ######
			###########################
			
			if not terminate:
				# all physics happens here
				# update dynamic arguments in place: ramp = m*t+b

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
			
			# update episodic return
			Return_j += R

			# update protocol and time
			protocol_inst.append(S_prime[0])
			t_inst.append(t_step*delta_t)
		
			# record action taken
			actions_taken.append(A)

			############################

			# calculate usage and alpha vectors: alpha_inf = eta
			u[theta_inds]*=(1.0-eta)
			u[theta_inds]+=1.0
			alpha = 1.0/(N_tilings*u[theta_inds])
			
			# Q learning update rule
			delta = R - Q[indA] # error in gradient descent
			# define error in time
			if t_step > 0:
				delta_TO = Q_old[indA] - np.sum(theta[theta_inds,t_step-1,indA],axis=0)
			else:
				delta_TO=0.0
			
			# update traces
			#e[theta_inds,t_step,indA] = alpha*(trace_fn(e[theta_inds,t_step,indA],alpha) - lmbda*E) 
			e[theta_inds,t_step,indA] = alpha*(np.ones_like(e[theta_inds,t_step,indA]) - lmbda*E) 
	
			# theta <-- theta + \alpha*[theta(t-1)\phi(S) - theta(t)\phi(S)]
			theta[theta_inds,t_step,indA] += alpha*delta_TO
			
			# check if S_prime is terminal or went out of grid
			if t_step == max_t_steps-1 or terminate: 
				# update theta
				theta += delta*e
				# set terminate variable to False
				terminate = False
				# go to next episode
				break

			
			# get set of features present in S_prime
			theta_inds = RL.find_feature_inds(S_prime,tilings,shift_tile_inds)
			# TO
			Q_old = np.sum(theta[theta_inds,t_step,:],axis=0)
			# non-TO
			Q = np.sum(theta[theta_inds,t_step+1,:],axis=0)

			# update theta and e
			delta += max(Q)
			theta += delta*e

			E = np.sum(e[theta_inds,t_step,indA],axis=0)
			e *= lmbda
			
			################################

			# S <- S_prime
			S[:] = S_prime[:]

		

		if greedy:
			return protocol_inst, t_inst  

		# save average return
		Return_ave[j] = 1.0/(j+1)*(Return_j + j*Return_ave[j-1])
		Return[j] = Return_j
		FidelitY[j] = fidelity
	

		# if greedy policy completes a full episode and if greedy fidelity is worse than inst one
		if len(actions_taken)==max_t_steps and fidelity-best_fidelity>1E-12:
			# update list of best actions
			best_actions[:] = actions_taken[:]
			# calculate best protocol and fidelity
			protocol_best, t_best = best_protocol(best_actions,hx_i,delta_t)
			t_vals, p_vals = t_best, protocol_best
			
			R_best=R
			best_fidelity=fidelity
			#best_fidelity = Fidelity(psi_i,H_fid,t_vals,delta_t,psi_f=psi_f,all_obs=False)[-1]
			
			#print best_fidelity, R_best

			theta = Learn_Policy(state_i,theta,tilings,dims,best_actions,R_best)
			#theta = Replay(50,RL_params,physics_params,theta,tilings,best_actions,R_best)
		
		# replay best encountered every 100 episodes
		if (j%40==0 and j!=0) and (R_best is not None) and beta_RL<1E12:
			print 'learned best encountered'
			theta = Learn_Policy(state_i,theta,tilings,dims,best_actions,R_best)
			#theta = Replay(50,RL_params,physics_params,theta,tilings,best_actions,R_best)


		#'''	
		if j%20 == 0:
			print "finished simulating episode {} with fidelity {} at hx_f = {}.".format(j+1,np.round(fidelity,3),S_prime[0])
			print 'best encountered fidelity is {}.'.format(np.round(best_fidelity,3))
		#'''

		#'''
		# plot protocols and learning rate
		if (j%500==0 and j!=0) or (np.round(fidelity,5) == 1.0):

			RL_params['beta_RL']=1E12
			RL_params['lmbda']=0.0
			RL_params['alpha_0']=0.0

			# fig file name params
			save = False #True
			save_vars = ['J','hz','hxi','hxf','Ei','Ef','Neps']
			save_vals = truncate([J,hz,hx_i,hx_f,E_i/L,E_f/L,j],2)
			save_params = "_L={}".format(L) + "".join(['_'+i+'='+k for i, k in zip(save_vars,save_vals) ])
			

			# calculate greedy fidelity
			Q_args = (RL_params,physics_params)
			Q_kwargs = {'theta':theta,'tilings':tilings}
			protocol_greedy, t_greedy = Q_learning(*Q_args,greedy=True,**Q_kwargs)

					
			# calculate inst fidelities of interpolated protocols
			t_vals, p_vals = t_inst, protocol_inst
			F_inst, E_inst, dE_inst, Sent_inst, Sd_inst = Fidelity(psi_i,H_fid,t_vals,delta_t,psi_f=psi_f,all_obs=True,Vf=Vf)

			t_vals, p_vals = t_greedy, protocol_greedy
			F_greedy, E_greedy, dE_greedy, Sent_greedy, Sd_greedy = Fidelity(psi_i,H_fid,t_vals,delta_t,psi_f=psi_f,all_obs=True,Vf=Vf)
			
			t_vals, p_vals = t_best, protocol_best
			F_best, E_best, dE_best, Sent_best, Sd_best = Fidelity(psi_i,H_fid,t_vals,delta_t,psi_f=psi_f,all_obs=True,Vf=Vf)

			
			# prepare plot data
			times = [t_inst,t_greedy,t_best]
			protocols = [protocol_inst,protocol_greedy,protocol_best]
			fidelities = [F_inst,F_greedy,F_best]
			energies = [E_inst,E_greedy,E_best]
			d_energies = [dE_inst,dE_greedy,dE_best]
			s_ents = [Sent_inst,Sent_greedy,Sent_best]
			s_ds = [Sd_inst,Sd_greedy,Sd_best]


			Data = np.zeros((7,max_t_steps))
			Data[0,:] = t_best
			Data[1,:] = protocol_best
			Data[2,:] = F_best
			Data[3,:] = E_best
			Data[4,:] = dE_best
			Data[5,:] = Sent_best
			Data[6,:] = Sd_best

			# plot data	
			user_input = raw_input("continue? (y or n) ") 
			if user_input=='y':
				# plot rewards
				#plot_rewards(N_episodes,Return_ave,Return,FidelitY,'rewards',save_params,save)
				# plot protocols
				plot_protocols(times,protocols,fidelities,'fidelity',save_params,save)
				#plot_protocols(times,protocols,energies,'energy',save_params,save)
				#plot_protocols(times,protocols,d_energies,'energy fluct.',save_params,save)
				#plot_protocols(times,protocols,s_ents,'ent. entropy',save_params,save)
				#plot_protocols(times,protocols,s_ds,'diag. entropy',save_params,save)		
				
				"""		
				# calculate approximate Q function
				etas = np.linspace(hx_limits[0],hx_limits[1],101)
				#etas = np.linspace(-1.0,1.0,101)
				Q_plot = RL.Q_greedy(etas,theta,tilings,shift_tile_inds,max_t_steps).T
				
				plot_Q(etas,t_best,-Q_plot,'Q_fn',save_params,save)
				"""

				if save:
					user_input = raw_input("save data? (y or n) ")
					if user_input=='y':
						args = (L,) + tuple( np.around([J,hz,hx_i,hx_f] ,2) )
						dataname  =  "best_L=%s_J=%s_hz=%s_hxi=%s_hxf=%s.txt"   %args
						np.savetxt(dataname,Data.T)

		

			RL_params['beta_RL']=beta_RL
			RL_params['lmbda']=lmbda
			RL_params['alpha_0']=alpha_0		
		#'''

	

	print "Calculating the Q function loop using Q-Learning took",("--- %s seconds ---" % (time.time() - start_time))


def best_protocol(best_actions,hx_i,delta_t):
	""" This function builds the best encounteres protocol from best_actions """
	s = hx_i
	protocol=[]
	t = [delta_t*_i for _i in range(len(best_actions))] #[0.0]
	for _i,a in enumerate(best_actions):
		s+=a
		protocol.append(s)
	return protocol, t


### only for plotting

def plot_Q(x,y,Q_plot,save_name,save_params,save=False):
	""" This function plots the Q function """
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	X,Y = np.meshgrid(list(x),list(y))

	ax.plot_surface(X, Y, Q_plot)

	ax.set_xlabel('$h_x$')
	ax.set_ylabel('$t$')
	ax.set_zlabel('Q')

	if save:
		save_str = save_name+save_name+'.png'
		plt.savefig(save_str)

	plt.show()

def plot_rewards(N_episodes,Return_ave,Return,FidelitY,save_name,save_params,save=False):
	""" This function plots the rewards vs episodes. """

	str_R = "$\\mathrm{episodic}$"
	str_Rave = "$\\mathrm{average}$"
	str_F = "$\\mathrm{fidelity}$"
	
	plt.plot(xrange(N_episodes),FidelitY,'g',linewidth=2.0,label=str_F)
	plt.plot(xrange(N_episodes),Return,'r-.',linewidth=0.5,label=str_R)
	plt.plot(xrange(N_episodes),Return_ave,'b',linewidth=2.0,label=str_Rave)
	

	#plt.xscale('log')

	plt.xlabel('$\\#\\ \\mathrm{episodes}$', fontsize=20)
	plt.ylabel('$\\mathrm{reward}$', fontsize=20)

	plt.legend(loc='lower right')
	plt.tick_params(labelsize=16)
	plt.grid(True)

	if save:
		save_str = save_name+save_params+'.png'
		plt.savefig(save_str)

	plt.show()

def plot_protocols(times,protocols,quantities,save_name,save_params,save=False):
	""" This function plots the protocols vs. time."""


	str_p = ["inst. protocol", "greedy protocol", "best protocol"]
	str_f = ["inst. fidelity", "greedy fidelity", "best fidelity"]
	

	if save_name=='fidelity':
		title_obs = 'F'
	elif save_name=='ent. entropy':
		title_obs = '\\mathcal{S}^\\mathrm{ent}'
	elif save_name=='diag. entropy':
		title_obs = '\\mathcal{S}^d'
	elif save_name=='energy':
		title_obs='\\mathcal{E}'
	elif save_name=='energy fluct.':
		title_obs='\\Delta\\mathcal{E}'

	igb = ['inst', 'greedy', 'best']
	str_c = ['b','r','g'] # str of colours
	titles=[]
	str_p=[]
	str_f=[]
	for _k in igb: 
		titles.append( title_obs + "_\\mathrm{" + _k + "}(t_f)=%s,\ " )

		str_p.append( _k + " protocol" )
		str_f.append( _k + " " + save_name )

	#titles = ["F_\\mathrm{inst}(t_f)=%s", ",\\ F_\\mathrm{greedy}(t_f)=%s", ",\\ F_\\mathrm{best}(t_f)=%s"]

	params = ()
	titlestr = "$"

	t_max,p_max,p_min =[],[],[]
	for j in xrange(len(times)):

		params += tuple( np.around( [quantities[j][-1]], 3) )
		titlestr += titles[j]

		plt.step(times[j],protocols[j],str_c[j],marker='.',linewidth=1,label=str_p[j])
		plt.plot(times[j],quantities[j],str_c[j]+'--',linewidth=1,label=str_f[j])

		t_max.append(times[j][-1])
		p_max.append( max( max(protocols[j]),max(quantities[j]) ) )
		p_min.append( min( min(protocols[j]),min(quantities[j])	) )
	
	titlestr += "$"
	titlestr = titlestr %(params)

	plt.xlabel('$t$', fontsize=20)
	plt.ylabel('$h_x(t)$', fontsize=20)

	plt.xlim([0,max(t_max)])
	plt.ylim([min(p_min)-0.5,max(p_max)+0.5])

	plt.legend(loc='best') #upper left
	plt.title(titlestr, fontsize=18)
	plt.tick_params(labelsize=16)
	plt.grid(True)

	if save:
		save_str = save_name+save_params+'.png'
		plt.savefig(save_str)

	pylab.show()

