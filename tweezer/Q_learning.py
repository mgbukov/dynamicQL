import numpy as np
import numpy.random as random
from scipy.sparse.linalg import expm_multiply
from numpy import exp

import descrete_SE,tweezer_hamiltonian
#from  descrete_SE import V_mat
from scipy.sparse import dia_matrix
from numpy import ones_like,array,vstack,float32

#import plot_data as plot
from quspin.operators import exp_op
from quspin.tools.measurements import ent_entropy

import time
import sys
import os
import cPickle

# make system update output files regularly
sys.stdout.flush()

# set pseudorandom generator
seed = random.randint(0,4294967295)
random.seed(seed)
#seed=random.seed(4210221767)

print("using seed={}".format(seed))


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

def explore_beta(t,m,b,T,beta_RL_const=1000.0):
	"""
	t: episode number/time
	m: slope of increase
	b: y intercept
	T: duration of ramp before zeroing
	"""
	if (t//T)%2==1:
		return beta_RL_const
	#elif (float(t)/40).is_integer():
	#	return 15.0
	else:
		return b + m/2.0*(float(t)/T - (t//T)/2.0) 
	

def best_protocol(best_actions,hx_i,delta_t):
	""" This function builds the best encounteres protocol from best_actions """
	protocol=np.zeros_like(best_actions)
	t = np.array([delta_t*_i for _i in range(len(best_actions))])
	S = hx_i
	for _i,A in enumerate(best_actions):
		S+=A
		protocol[_i]=S
	return protocol, t


def greedy_protocol(theta,tilings,actions,hx_i,delta_t,max_t_steps,h_field):
	""" This function builds the best encounteres protocol from best_actions """
	protocol=np.zeros((max_t_steps,),dtype=np.float64)
	t = np.array([delta_t*_i for _i in range(max_t_steps)])
	S = np.array([hx_i])
	
	N_tilings = tilings.shape[0]
	N_tiles = tilings.shape[1]
	
	# preallocate theta_inds
	theta_inds_zeros=np.zeros((N_tilings,),dtype=int)
	
	for t_step in range(max_t_steps):

		avail_inds = np.argwhere((S[0]+np.array(actions)<=h_field[-1])*(S[0]+np.array(actions)>=h_field[0])).squeeze()
		avail_actions = actions[avail_inds]

		# calculate Q(s,a)
		theta_inds=find_feature_inds(tilings,S,theta_inds_zeros)
		Q = np.sum(theta[theta_inds,t_step,:],axis=0)
		# find greedy action
		A_greedy=avail_actions[np.argmax(Q[avail_inds])]

		S[0]+=A_greedy
		protocol[t_step]=S

	return protocol, t


def Learn_Policy(state_i,best_actions,R,theta,tilings,actions,ep=None):

	N_tilings = tilings.shape[0]
	N_tiles = tilings.shape[1]
	
	# preallocate theta_inds
	theta_inds_zeros=np.zeros((N_tilings,),dtype=int)

	S = state_i.copy()

	for t_step, A in enumerate(best_actions):
		# calculate state-action indices
		indA = np.searchsorted(actions,A)
		
		theta_inds=find_feature_inds(tilings,S,theta_inds_zeros)
		
		# check if max learnt
		if max(theta[theta_inds,t_step,:].ravel())>R/N_tilings and R>0:
			Q = np.sum(theta[theta_inds,t_step,:],axis=0)
			indA_max=np.argmax(Q)

			if max(Q) > 1E-13:
				theta[theta_inds,t_step,:]*=R/Q[indA_max]
			
		# calculate theta function
		theta[theta_inds,t_step,indA] = (R+1E-2)/(N_tilings)
		
		S+=A

	#print 'force-learned best encountered policy'
	return theta


def find_feature_inds(tilings,S,theta_inds):
	S=[S]
	for _k, tiling in enumerate(tilings): # cython-ise this loop as inline fn!
		idx = tiling.searchsorted(S)
		idx = np.clip(idx, 1, len(tiling)-1)
		left, right = tiling[idx-1], tiling[idx]
		idx -= S - left < right - S
		theta_inds[_k]=idx[0]+_k*tilings.shape[1]
	return theta_inds

# constructs a diagonal matrix with the function as the diagonal
def V_mat(x,V_func,V_args,dtype=float32):
	n = x.shape[0]
	ff = V_func(x,*V_args)
	return  dia_matrix((ff,[0]),shape=(n,n),dtype=dtype)

def tweezers(x,x0,d0):
	return exp(-32.0*(x-x0)**2)*d0 - 130*exp(-32.0*(x-0.55)**2)





def Q_learning(N,N_episodes,alpha_0,eta,lmbda,beta_RL_i,beta_RL_inf,T_expl,m_expl,N_tilings,N_tiles,state_i,h_field,dh_field,bang,
			   L,max_t_steps,delta_time,x,x0,d_tweezer,psi_i,psi_f,
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

	# prealocate kinetic energy
	H_kin=0.5*descrete_SE.laplace_mat(x,x[1]-x[0])

	# preallocate physical state
	psi = np.zeros_like(psi_i)

	##### RL quantities	#####

	# define actions
	if bang:
		pos_actions=[8.0]; a_str='_bang';
	else:
		pos_actions=[0.01,0.02,1.1]; a_str='_cont';
	
	neg_actions=[-i for i in pos_actions]
	actions = np.sort(neg_actions + [0.0] + pos_actions)
	
	# calculate unitaries dictionary
	U_dict=tweezer_hamiltonian.Unitaries_1(H_kin,x,delta_time,d_tweezer,pos_actions,[min(x),max(x)])

	N_actions = len(actions)
	
	if theta is None:
		theta=np.zeros((N_tiles*N_tilings,max_t_steps,N_actions), dtype=np.float64)
	theta_old=theta.copy()
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
	best_R = -1.0 # best encountered fidelity
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
		theta_inds=find_feature_inds(tilings,S[0],theta_inds_zeros)
		Q = np.sum(theta[theta_inds,0,:],axis=0) # for each action at time t_step=0

		# preallocate physical quantties
		psi[:] = psi_i[:] # quantum state at time
		
		# taken encountered and taken
		actions_taken = np.zeros((max_t_steps,),dtype=np.float64)
		

		#define beta
		beta_RL = explore_beta(ep,m_expl,beta_RL_i,T_expl,beta_RL_const=beta_RL_inf)

		explored=False
		# generate episode
		for t_step in xrange(max_t_steps): 

			# calculate available actions from state S
			avail_inds = np.argwhere((S[0]+np.array(actions)<=h_field[-1])*(S[0]+np.array(actions)>=h_field[0])).squeeze()
			avail_actions = actions[avail_inds]

			if beta_RL < beta_RL_inf: #20.0
				if ep%2==0:
					A_greedy = avail_actions[random.choice(np.argwhere(Q[avail_inds]==np.amax(Q[avail_inds])).ravel() ) ]
				else:
					A_greedy=best_actions[t_step]
			else:
				A_greedy = avail_actions[random.choice(np.argwhere(Q[avail_inds]==np.amax(Q[avail_inds])).ravel() ) ]
				

			if beta_RL < beta_RL_inf:
				# choose a random action
				P = np.exp(beta_RL*Q[avail_inds])
				A = avail_actions[np.searchsorted(np.cumsum(P/np.sum(P)),random.uniform(0.0,1.0))]
				
				# reset traces if A is exploratory
				if abs(A - A_greedy) > np.finfo(A).eps:
					e *= 0.0
			else:
				A = A_greedy

			# find the index of A
			indA = np.searchsorted(actions,A)
			
			# record action taken
			actions_taken[t_step]=A

			# take action A, return state S_prime and actual reward R
			################################################################################
			######################    INTERACT WITH ENVIRONMENT    #########################
			################################################################################

			# define new state
			S_prime = S.copy()
			S_prime[0] += A
			"""
			# calculate new field value 
			if t_step == max_t_steps-1:
				S_prime[0] = -0.55
			"""		
			
			
			
			# all physics happens here
			x_tweezer = S_prime[0] # update position of tweezer

			"""
			# build new Hamiltonian
			#H=H_kin+V_mat(x,tweezers,(x_tweezer,d_tweezer))
			# evolve wave function
			psi = expm_multiply(-1j*delta_time*H,psi) #exp_H.dot(psi)
			"""
			psi = U_dict[int(np.rint((x_tweezer - min(x))/min(pos_actions)))].dot(psi)

			# assign reward
			R *= 0.0
			if t_step == max_t_steps-1:
				# calculate final fidelity and give it as a reward
				#EGS = H.eigsh(k=1,which='SA',maxiter=1E10,return_eigenvectors=False).squeeze()
				R += abs(psi.conj().dot(psi_f))**2 #-(H.matrix_ele(psi,psi).real-EGS) #-ent_entropy(psi,H.basis)['Sent'] #
				
				#print(R)

			################################################################################
			################################################################################
			################################################################################

			# calculate usage and alpha vectors: alpha_inf = eta
			u[theta_inds]*=(1.0-eta)
			u[theta_inds]+=1.0
			alpha = 1.0/(N_tilings*u[theta_inds])
			
			# Q learning update rule; GD error in time t
			delta_t = R - Q[indA] # error in gradient descent
			# TO
			Q_old = theta[theta_inds,t_step,indA].sum()
			
			# update traces
			e[theta_inds,t_step,indA] = alpha*fire_trace

			# check if S_prime is terminal or went out of grid
			if t_step == max_t_steps-1: 
				# update theta
				theta += delta_t*e
				# GD error in field h
				delta_h = Q_old - theta[theta_inds,t_step,indA].sum()
				theta[theta_inds,t_step,indA] += alpha*delta_h
				# go to next episode
				break

			# get set of features present in S_prime
			theta_inds_prime=find_feature_inds(tilings,S_prime,theta_inds_zeros)

			# t-dependent Watkin's Q learning
			Q = np.sum(theta[theta_inds_prime,t_step+1,:],axis=0)

			# update theta
			delta_t += np.max(Q)
			theta += delta_t*e

			# GD error in field h
			delta_h = Q_old - theta[theta_inds,t_step,indA].sum()
			theta[theta_inds,t_step,indA] += alpha*delta_h

			# update traces
			e[theta_inds,t_step,indA] -= alpha*e[theta_inds,t_step,indA].sum()
			e *= lmbda

			################################
			# S <- S_prime
			S[:] = S_prime[:]
			theta_inds[:]=theta_inds_prime[:]
	
		# if greedy policy completes a full episode and if greedy fidelity is worse than inst one
		if R-best_R > 1E-12:
			print("best encountered fidelity is {}".format(np.around(R,4)) )
			# update list of best actions
			best_actions = actions_taken[:]
			# best reward and fidelity
			best_R = R
			# learn policy
			#if beta_RL<20.0:
			theta = Learn_Policy(state_i,best_actions,best_R,theta,tilings,actions)

		# force-learn best encountered every 40 episodes
		if ( (ep+1)%(2*T_expl)-T_expl==0 and ep not in [0,N_episodes-1] ):# and beta_RL<20.0:
			theta = Learn_Policy(state_i,best_actions,best_R,theta,tilings,actions,ep=ep)
		elif (ep//T_expl)%2==1 and abs(R-best_R)>1E-12:
			theta = Learn_Policy(state_i,best_actions,best_R,theta,tilings,actions,ep=ep)


		#"""
		# check if Q-function converges
		print ep, "beta_RL,R,d_theta:",beta_RL,R,np.max(abs(theta.ravel() - theta_old.ravel() ))
		theta_old=theta.copy()
		#"""

			
		# record average return
		Return_ave[ep] = 1.0/(ep+1)*(R + ep*Return_ave[ep-1])
		Return[ep] = R
		Fidelity_ep[ep] = R


		if (ep+1)%(2*T_expl) == 0:
			print "finished simulating episode {} with fidelity {} at hx_f = {}.".format(ep+1,np.round(R,5),S_prime[0])
			print 'best encountered fidelity is {}.'.format(np.round(best_R,5))
			#print 'current inverse exploration tampeature is {}.'.format(np.round(beta_RL,3))
			
			# calculate best protocol and fidelity
			protocol_best,t_best = best_protocol(best_actions,state_i[0],delta_time)
			protocol_greedy,t_greedy = greedy_protocol(theta,tilings,actions,state_i[0],delta_time,max_t_steps,h_field)
			
			print np.around(protocol_best,3)
			#print protocol_greedy

	# save data
	Data_fid = np.zeros((N_episodes,3))
	
	Data_fid[:,0] = Fidelity_ep
	Data_fid[:,1] = Return
	Data_fid[:,2] = Return_ave
	#
	Data_protocol = np.zeros((max_t_steps,3))

	Data_protocol[:,0] = t_best
	Data_protocol[:,1] = protocol_best
	Data_protocol[:,2] = protocol_greedy

	# define parameter-dependent part of file name
	args = (N,N_episodes,max_t_steps,L) + tuple( truncate([x0,d_tweezer] ,2) )
	data_params = "_N=%s_Nep=%s_T=%s_L=%s_x0=%s_dtweezer=%s"   %args
	data_params+=a_str

	### define save directory for data
	# read in local directory path
	str1=os.getcwd()
	str2=str1.split('\\')
	n=len(str2)
	my_dir = str2[n-1]
	# create directory if non-existent
	save_dir = my_dir+"/data"
	if not os.path.exists(save_dir):
	    os.makedirs(save_dir)
	save_dir="data/"

	if save:
		# display full strings
		np.set_printoptions(threshold='nan')

		# as txt format
		dataname  = save_dir + "RL_data"+data_params+'.txt'
		np.savetxt(dataname,Data_fid)

		dataname  = save_dir + "protocol_data"+data_params+'.txt'
		np.savetxt(dataname,Data_protocol)
		# save as pickle
		dataname  = save_dir + "theta_data"+data_params+'.pkl'
		cPickle.dump(theta, open(dataname, "wb" ) )
		#cPickle.load(open(dataname, "rb" ))

		dataname  = save_dir + "tilings_data"+data_params+'.pkl'
		cPickle.dump(tilings, open(dataname, "wb" ) )

		RL_params = {"N":N,
					 "N_episodes":N_episodes,
					 "alpha_0":alpha_0,
					 "eta":eta,
					 "lmbda":lmbda,
					 "beta_RL_i":beta_RL_i,
					 "beta_RL_inf":beta_RL_inf,
					 "T_expl":T_expl,
					 "m_expl":m_expl,
					 "N_tilings":N_tilings,
					 "N_tiles":N_tiles,
					 "state_i":state_i,
					 "h_field":h_field,
					 "dh_field":dh_field,
					 "seed":seed,
					 }
		dataname  = save_dir + "RL_params_data"+data_params+'.pkl'
		cPickle.dump(RL_params, open(dataname, "wb" ) )

		phys_params = {"L":L,
					   "max_t_steps":max_t_steps,
					   "delta_time":delta_time,
					   "x":x,
					   "x0":x0,
					   "d_tweezer":d_tweezer,
					   "psi_i":psi_i,
					   "psi_f":psi_f,
					   } 
		dataname  =  save_dir + "phys_params_data"+data_params+'.pkl'
		cPickle.dump(phys_params, open(dataname, "wb" ) )

	"""
	# create plots
	plot.plot_rewards(Fidelity_ep,Return,Return_ave,'RL_stats',data_params)

	plot.observables(L,t_best,protocol_best,hx_i,hx_f,J,hz,data_params+'_best')
	plot.observables(L,t_best,protocol_greedy,hx_i,hx_f,J,hz,data_params+'_greedy')
	"""



		
