import numpy as np
import functools

import random
random.seed()

# define actions
#pos_actions=list(0.2*np.asarray([0.01,0.02,0.05,0.1,0.2,0.5,1.0]))
pos_actions=[0.01,0.02,0.05,0.1,0.2,0.5,1.0]
def all_actions():
	neg_actions=[]
	[neg_actions.append(-i) for i in pos_actions]
	return sorted(pos_actions + [0.0] + neg_actions)

actions = all_actions()

def actions_length():
	return len(actions)

def avail_actions(state,time_steps_left,hx_f,ind=False):
	av_actions = actions
	if ind:
		return av_actions, [actions.index(i) for i in av_actions]
	else:
		return av_actions
 

# define tilings
def gen_tilings(Vars,dVars,N_tilings):
	"""
	v: list of state Vars
	dv: list of grid step sizes
	"""
	var0 = Vars[0]
	#var1 = Vars[1]
	#var2 = Vars[2]

	#tiling = np.array( [[i,j,k] for i in var0 for j in var1 for k in var2] )
	tiling = np.array( [[i] for i in var0] )

	tilings = [tiling + np.random.uniform(0,dVars,len(Vars)) for j in xrange(N_tilings)]

	return tilings


def find_tile(x,tiling):
	"""
	Calculates the index of the 'tiling's element closest to 'x'.
	tiling: array of grid positions
	x: variable of the same type as the elements of tiling
	"""
	return np.argmin(np.linalg.norm(tiling - x, axis=1))

def find_feature_inds(S,tilings,N_tiles):

	# find indices of S in tilings
	inds = map(functools.partial(find_tile, S), tilings)
	# add up a shift to get the index position of vector theta
	return [inds[j] + j*N_tiles for j in xrange(len(inds))]



def E_traces_acc(E,alpha=None):
	return E + 1.0
def E_traces_dutch(E,alpha):
	E *= 1.0 - alpha
	return E + 1.0
def E_traces_repl(E,alpha=None):
	return np.ones(E.shape)

def Q_greedy(Vars,theta,tilings,N_tiles,max_t_steps):
	var0 = Vars

	Q = np.zeros((len(var0),max_t_steps),dtype=np.float64)
	grid_states = [np.array([i]) for i in var0]

	for S in grid_states:

		i_s = np.argmin(abs(var0-S[0]) )
		
		theta_inds = find_feature_inds(S,tilings,N_tiles)

		A = np.max( np.sum(theta[theta_inds,:,:],axis=0), axis=1)

		Q[i_s,:] = A

	return Q




