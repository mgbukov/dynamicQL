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

#import cTimer

def truncate(f, n):
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

"""
def Fidelity(psi_i,H_fid,t_vals,basis=None,psi_f=None,all_obs=False):
	''' This function calculates the physical quantities given a time-dep Hamiltonian H_fid.
		If psi_f is not given, then it returns the fidelity in the instantaneous eigenstate, otherwise
		--- the fidelity in the final state. 
	'''
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
"""

def Fidelity(psi_i,H_fid,t_vals,delta_t,basis=None,psi_f=None,all_obs=False,Vf=None):
	""" This function calculates the physical quantities given a time-dep Hamiltonian H_fid.
		If psi_f is not given, then it returns the fidelity in the instantaneous eigenstate, otherwise
		--- the fidelity in the final state. """
	# get sysstem size
	L = basis.L
	# evolve state
	#psi_t = H_fid.evolve(psi_i,t_vals[0],t_vals,iterate=True,atol=1E-12,rtol=1E-12)


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
		Sd=[]

	psi=psi_i.copy()
	for i, t_i in enumerate(t_vals):

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

			EGS,_ = H_fid.eigsh(time=t_vals[i],k=2,which='BE',maxiter=1E10,return_eigenvectors=False)
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

	
	# re-set parameters to calculate greedy polocy
	RL_params['eps']=0.0
	RL_params['lmbda']=0.0
	RL_params['alpha_0']=0.0

	######################################################################

	
	# get dimensions
	N_tilings, N_lintiles, N_vars = dims
	N_tiles = N_lintiles**N_vars
	N_actions = RL.actions_length()

	if not greedy:
		# prealloate theta
		theta = np.zeros((N_tiles*N_tilings,max_t_steps,N_actions), dtype=np.float64)
		# calculate tilings
		tilings = RL.gen_tilings(Vars,dVars,N_tilings)

		
	# eta limits
	hx_limits = [Vars[0][0],Vars[0][-1]]

	# pre-allocate traces variable
	e = np.zeros(theta.shape, dtype=np.float64)
	#e = np.zeros((N_tiles*N_tilings,N_actions), dtype=np.float64)
	
	

	if not greedy:
		# pre-allocate usage vector
		u0 = 1.0/alpha_0*np.ones((N_tiles*N_tilings,), dtype=np.float64)
		#u0 = 1.0/alpha_0*np.ones(theta.shape, dtype=np.float64)
	else:
		u0 = np.inf*np.ones((N_tiles*N_tilings,), dtype=np.float64)
		#u0 = np.inf*np.ones(theta.shape, dtype=np.float64)
	v0 = np.zeros(u0.shape, dtype=np.float64)

	# set terminate episode variable for wandering off the eta grid
	terminate = False
	# define all actions
	actions = RL.all_actions()

	#### physical quantities

	# define ED Hamiltonian H(t)
	#m=0.0
	b=hx_i
	#lin_fun = lambda t: m*t + b
	lin_fun = lambda t: b
	# define Hamiltonian
	H, basis = Hamiltonian.Hamiltonian(L,fun=lin_fun,**{'J':J,'hz':hz})
	# defien Hamiltonian for interpolated fidelity
	t_vals, p_vals = [0.0,0.0], [0.0,0.0]
	#protocol_fun = lambda t: np.interp(t, t_vals, p_vals)
	def step_protocol(t):
		return p_vals[np.argmin( abs(np.asarray(t_vals)-t) )]
	#H_fid,_ = Hamiltonian.Hamiltonian(L,fun=protocol_fun,basis=basis,**{'J':J,'hz':hz})
	H_fid,_ = Hamiltonian.Hamiltonian(L,fun=step_protocol,basis=basis,**{'J':J,'hz':hz})

	# calculate final basis
	_,Vf = H.eigh(time=0.0)
	b=hx_i

	# set trace function
	trace_fn = eval('RL.E_traces_'+traces)
	
	# average reward
	Return_ave = np.zeros((N_episodes,1),dtype=np.float64)
	Return = Return_ave.copy()
	FidelitY = Return_ave.copy()

	# initialise best fidelity
	best_fidelity = 0.0
	best_actions=[random.choice(actions) for j in range(max_t_steps)]

		
	# loop over episodes
	avail_actions = actions
	a_inds = [i for i in range(len(avail_actions))]
	for j in xrange(N_episodes):
		# set traces to zero
		e *= 0.0
		# set initial usage vector
		u = u0.copy()
		# set initial aux v vector
		v = v0.copy()

		# set initial state of episode
		S = state_i.copy()

		
		"""
		# get set of available actions and their indices
		#avail_actions, a_inds = RL.avail_actions(S,max_t_steps,hx_f,ind=True)
		# get theta(t-1)phi(S), i.e. store Q values as a list of length a_inds	 
		#Q = [theta[theta_inds,0,k].sum() for k in a_inds]
		"""
		
		# get set of features present in S
		theta_inds = RL.find_feature_inds(S,tilings,N_tiles)
		Q = np.sum(theta[theta_inds,0,:],axis=0)
		E=0.0
		Q_old=[0.0 for i in actions]
		# preallocate physical quantties
		psi = psi_i.copy() # #	quantum state at time

		inst_fidelity = 0.0
		fidelity = float('nan')
		protocol_inst = [hx_i]
		t_inst = [0.0]
		
		# calculate sum of all rewards
		Return_j = 0.0

		# taken encountered and taken
		actions_taken = []
		
		#eps = 0.1/np.sqrt(j+1)
		#eps = 0.1/np.log(j+2)
		#"""
		N_explore=100
		if j > N_explore:
			eps=0.0
		else:
			eps=0.1
		#"""
		# generate episode
		for t_step in xrange(max_t_steps): #
			

			# calculate greedy action(s) wrt behavior policy
			if not greedy:
				A_star = best_actions[t_step] 
				#A_star = avail_actions[random.choice( np.argwhere(Q==np.amax(Q)).ravel() )] 
			else:
				A_star = avail_actions[random.choice( np.argwhere(Q==np.amax(Q)).ravel() )]
			#print Q
			#print A_star, avail_actions[np.argmax(Q)], np.argmax(Q)
			if random.uniform(0,1) <= 1.0 - eps:
				A = A_star
			else:
				A = random.choice(list(set(avail_actions) - set([A_star]) )) #random.choice(avail_actions)
			
			# find the index of A
			avail_indA = avail_actions.index(A)
			indA = a_inds[avail_indA]

			
			# reset traces and calculate probability under beh policy
			if abs(A - A_star) > np.finfo(A).eps:
				print 'traces reset'
				e *= 0.0
				beh_policy = eps/len(a_inds)
			else:
				beh_policy = eps/len(a_inds) + 1.0-eps

			# calculate probability of taking A under target policy
			if A not in [avail_actions[a] for a in np.argwhere(Q==np.amax(Q))]:
				tgt_policy = 0.1/len(a_inds) #eps/len(a_inds)
			else:
				tgt_policy = 0.1/len(a_inds) + 1.0-0.1 #eps/len(a_inds) + 1.0-eps

			# calculate importance sampling ratio
			#rho = tgt_policy/beh_policy
			rho = min(1.0,tgt_policy/beh_policy)
			#rho=1.0
			#'''
			if rho > 1.0:
				#rho = 0.0 
				print 'rho > 1'
				#user_input = raw_input("continue? (y or n) ")
			#'''
			# take action A, observe state S_prime and reward R
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
			R = 0.0

			# penalise and set to terminate if S_prime is outside of grid 
			if S_prime[0] < hx_limits[0] or S_prime[0] > hx_limits[1]:
				R += -1.0
				terminate = True

			###########################
			######  Environment  ######
			###########################

			if not terminate:

				# update dynamic arguments in place: ramp = m*t+b
				#m = (S_prime[0]-S[0])/delta_t
				#b = S[0] - m*t_inst[-1]
				#psi = H.evolve(psi,t_inst[-1],t_inst[-1]+delta_t,atol=1E-9,rtol=1E-9)
				b = S_prime[0]
				psi = exp_op(H(time=t_step*delta_t),a=-1j*delta_t).dot(psi)

				'''
				### enable these lines if instantaneous fidelity is needed
				# sparse
				_,psi_inst = _sla.eigsh(H,time=t_inst[-1]+delta_t,k=1,sigma=-100)[1] 
				psi_inst = psi_inst.squeeze()
				# calculate instantaneous fidelity
				inst_fidelity += abs( psi.conj().dot(psi_inst) )**2
				# give inst reward
				R += (1-mu)*inst_fidelity/max_t_steps
				'''

				# assign reward
				if t_step == max_t_steps-1:
					# calculate final fidelity
					fidelity = abs( psi.conj().dot(psi_f) )**2
					# reward
					R += mu*fidelity
				
			################################################################################
			################################################################################
			################################################################################
			
			# update episodic return
			Return_j += R

			# update protocol and time
			protocol_inst.append(S_prime[0])
			t_inst.append((t_step+1)*delta_t)
		
			# record action taken
			actions_taken.append(A)


			############################

			# calculate usage and alpha vectors
			u[theta_inds] *= (1.0-eta)
			u += (rho-1.0)*gamma*lmbda*v
			u[theta_inds] += rho - (rho-1.0)*gamma*lmbda*eta*v[theta_inds]
			# update aux v vector
			v *= gamma*lmbda*rho
			v[theta_inds] *= 1.0-eta
			v[theta_inds] += rho

			with np.errstate(divide='ignore'):
				alpha = 1.0/(N_tilings*u[theta_inds])
				alpha[u[theta_inds]<1E-12] = 1.0

			
			#"""
			if max(alpha)*rho > 1.0/N_tilings:
				#print "ALPHA", rho*alpha_0/N_tilings, rho*max(alpha), rho*min(alpha)
				print "alpha exiting.. at ep {} and step {} with rho={}".format(j,t_step,rho)
				exit()
			#"""

			#"""
			if any(Q>3.0):
				s = hx_i
				t_step=0
				#print "reward is", R 
				for a in actions_taken:
					theta_inds = RL.find_feature_inds(s,tilings,N_tiles)
					print s, np.sum(theta[theta_inds,t_step,:],axis=0)
					t_step+=1
					s+=a

				print 'exiting at episode {}'.format(j)
				exit()
			
			# Q learning update rule
			delta = R - Q[avail_indA]
			delta_TO = Q_old[avail_indA] - np.sum(theta[theta_inds,t_step,indA],axis=0)
			#print 'DELTA', delta_TO

			#user_input = raw_input("save data? (y or n) ")
			
			# update traces
			#e[theta_inds,t_step,indA] = rho*alpha*trace_fn(e[theta_inds,t_step,indA],alpha)
			e[theta_inds,t_step,indA] = rho*alpha*(trace_fn(e[theta_inds,t_step,indA],alpha) - gamma*lmbda*rho*E) 
	
			# theta <-- theta + \alpha*[theta(t-1)\phi(S) - theta(t)\phi(S)]
			#theta[theta_inds,t_step,indA] += rho*alpha*delta_TO
			
			# check if S_prime is terminal or went out of grid
			if t_step == max_t_steps-1 or terminate: 
				# update theta
				theta += delta*e
				# set terminate variable to False
				terminate = False
				# go to next episode
				print 'REWARD', R 
				break

			
			# get set of features present in S_prime
			theta_inds = RL.find_feature_inds(S_prime,tilings,N_tiles)
			"""
			# get set of available actions and their indices
			#avail_actions, a_inds = RL.avail_actions(S_prime,time_steps_left,hx_f,ind=True)
			# get theta(t)\phi(S'), i.e. store Q values as a list of length a_inds
			Q = [theta[theta_inds,t_step,k].sum() for k in a_inds]
			"""
			# TO
			Q_old = np.sum(theta[theta_inds,t_step,:],axis=0)
			# non-TO
			Q_prime = np.sum(theta[theta_inds,t_step+1,:],axis=0)

			"""
			if t_step in [6,11] and not greedy:
				print 'time/states', t_step, S, S_prime
				print 'A_star, A:', A_star, A
				print 'Q', Q
				print 'Q_prime', Q_prime				
				print 'delta',delta
				print 'final_delta', delta + gamma*max(Q_prime)
				print 'traces Spime', np.round( np.sum(  e[ RL.find_feature_inds(S_prime,tilings,N_tiles) ,t_step ,: ],axis=0), 3)
				print 'traces S', np.round( np.sum(  e[ RL.find_feature_inds(S,tilings,N_tiles) ,t_step ,: ],axis=0), 3)
				print 'rho, max_alpha', rho, max(alpha)
				print 'sum traces', np.round( np.sum(e), 3)
			
			"""
			# update theta and e

			delta += gamma*max(Q_prime)
			#delta=min(0.5,delta)
			theta += delta*e
			
			E = np.sum(e[theta_inds,t_step,indA],axis=0)
			e *= gamma*lmbda*rho
			
			################################


			#"""
			#print 'glr', gamma*lmbda*rho
			#print "state, action", S[0], actions[indA], S_prime[0]
			#print 'final delta', delta #, delta_TO
			#print 'Q and Q_prime', Q, Q_prime
			#print 'fucking elmt', np.round( np.sum(  theta[ RL.find_feature_inds(hx_i,tilings,N_tiles) ,0 ,: ],axis=0), 3)
			#print 'traces', np.round( np.sum(  e[ RL.find_feature_inds(hx_i,tilings,N_tiles) ,0 ,: ],axis=0), 3), gamma*lmbda*rho
			#"""

			# S <- S_prime
			S = S_prime[:]
			Q = Q_prime[:]

			
			if any(Q_prime>3.0):
				print 'post delta', delta, delta_TO
				print 'Q_prime', Q_prime

				t_step=0
				s = hx_i
				print "reward is", R 
				for a in actions_taken:
					theta_inds = RL.find_feature_inds(s,tilings,N_tiles)
					print np.sum(theta[theta_inds,t_step,:],axis=0), s
					t_step+=1
					s+=a

				print 'prime exiting at episode {}'.format(j)
				exit()


		#print '++++++++'
		#exit()

		if j > 2*N_explore:
			exit()

		print '_____________', j
		s = hx_i
		t_step=0
		print "reward is", R 
		for a in actions_taken:
			theta_inds = RL.find_feature_inds(s,tilings,N_tiles)
			"""
			print 'state', s
			print 'traces'
			print np.sum( e[theta_inds,t_step,:], axis=0)
			print '++++++'
			print 'thetas'
			"""
			print np.round( np.sum(theta[theta_inds,t_step,:],axis=0), 3), s
			t_step+=1
			s+=a
		print '_____________'


			
		if greedy:
			return protocol_inst, t_inst  

		# save average return
		Return_ave[j] = 1.0/(j+1)*(Return_j + j*Return_ave[j-1])
		Return[j] = Return_j
		FidelitY[j] = fidelity
	
	
		# if greedy policy completes a full episode and if greedy fidelity is worse than inst one
		if len(actions_taken)==max_t_steps and fidelity>best_fidelity and R >= 0:
			#print fidelity, best_fidelity
			# update list of best actions
			best_actions = actions_taken[:]
			# calculate best protocol and fidelity
			protocol_best, t_best = best_protocol(best_actions,hx_i,delta_t)
			t_vals, p_vals = t_best, protocol_best
			#best_fidelity = Fidelity(psi_i,H_fid,[t_vals[0],t_vals[-1]],basis=basis,psi_f=psi_f,all_obs=False)[-1]
			best_fidelity = Fidelity(psi_i,H_fid,t_vals,delta_t,basis=basis,psi_f=psi_f,all_obs=False)[-1]
		
		'''	
		if j%20 == 0:
			print "finished simulating episode {} with fidelity {} at hx_f = {}.".format(j+1,np.round(fidelity,3),S_prime[0])
			print 'best encountered fidelity is {}.'.format(np.round(best_fidelity,3))
		'''

		#'''
		# plot protocols and learning rate
		if (j%500==0 and j!=0) or (np.round(fidelity,5) == 1.0):

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
			F_inst, E_inst, dE_inst, Sent_inst, Sd_inst = Fidelity(psi_i,H_fid,t_vals,delta_t,basis=basis,psi_f=psi_f,all_obs=True,Vf=Vf)

			t_vals, p_vals = t_greedy, protocol_greedy
			F_greedy, E_greedy, dE_greedy, Sent_greedy, Sd_greedy = Fidelity(psi_i,H_fid,t_vals,delta_t,basis=basis,psi_f=psi_f,all_obs=True,Vf=Vf)
			
			t_vals, p_vals = t_best, protocol_best
			F_best, E_best, dE_best, Sent_best, Sd_best = Fidelity(psi_i,H_fid,t_vals,delta_t,basis=basis,psi_f=psi_f,all_obs=True,Vf=Vf)

			# prepare plot data
			times = [t_inst,t_greedy,t_best]
			protocols = [protocol_inst,protocol_greedy,protocol_best]
			fidelities = [F_inst,F_greedy,F_best]
			energies = [E_inst,E_greedy,E_best]
			d_energies = [dE_inst,dE_greedy,dE_best]
			s_ents = [Sent_inst,Sent_greedy,Sent_best]
			s_ds = [Sd_inst,Sd_greedy,Sd_best]


			Data = np.zeros((7,max_t_steps+1))
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
				
				#"""		
				# calculate approximate Q function
				etas = np.linspace(hx_limits[0],hx_limits[1],101)
				#etas = np.linspace(-1.0,1.0,101)
				Q_plot = RL.Q_greedy(etas,theta,tilings,N_tiles,max_t_steps).T
				
				plot_Q(etas,t_best[:-1],-Q_plot,'Q_fn',save_params,save)
				#"""

				if save:
					user_input = raw_input("save data? (y or n) ")
					if user_input=='y':
						args = (L,) + tuple( np.around([J,hz,hx_i,hx_f] ,2) )
						dataname  =  "best_L=%s_J=%s_hz=%s_hxi=%s_hxf=%s.txt"   %args
						np.savetxt(dataname,Data.T)

		#'''

	

	print "Calculating the Q function loop using Q-Learning took",("--- %s seconds ---" % (time.time() - start_time))


def best_protocol(best_actions,hx_i,delta_t):
	""" This function builds the best encounteres protocol from best_actions """
	s = hx_i
	protocol=[hx_i]
	t = [0.0]
	for a in best_actions:
		s+=a
		t.append(t[-1]+delta_t)
		protocol.append(s)
	return protocol, t


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

		plt.plot(times[j],protocols[j],str_c[j],linewidth=1,marker='.',label=str_p[j])
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

	plt.legend(loc='upper left')
	plt.title(titlestr, fontsize=18)
	plt.tick_params(labelsize=16)
	plt.grid(True)

	if save:
		save_str = save_name+save_params+'.png'
		plt.savefig(save_str)

	pylab.show()