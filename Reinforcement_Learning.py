import numpy as np
import functools
from operator import add

import random
random.seed()

# define actions
pos_actions=[0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0]
#pos_actions=[2.0]
def all_actions():
	neg_actions=[]
	[neg_actions.append(-i) for i in pos_actions]
	return sorted(pos_actions + [0.0] + neg_actions)


def gen_tilings(h_field,dh_field,N_tilings):
	return [h_field + np.random.uniform(0.0,dh_field,1) for j in xrange(N_tilings)]


def find_tile(x,tiling):
	"""
	Calculates the index of the 'tiling's element closest to 'x'.
	tiling: array of grid positions
	x: variable of the same type as the elements of tiling
	"""
	idx = tiling.searchsorted(x)
	idx = np.clip(idx, 1, len(tiling)-1)
	left = tiling[idx-1]
	right = tiling[idx]
	idx -= x - left < right - x
	
	return idx[0] 
	#return np.argmin(np.linalg.norm(np.array([ [i] for i in tiling]) - x, axis=1)) 

def find_feature_inds(S,tilings,shift_tile_inds):
	# find indices of S in tilings
	inds = map(functools.partial(find_tile, S), tilings)
	# add up a shift to get the index position of vector theta
	return map(add,inds,shift_tile_inds)

# used for plotting
def Q_greedy(Vars,theta,tilings,shift_tile_inds,max_t_steps):
	var0 = Vars

	Q = np.zeros((len(var0),max_t_steps),dtype=np.float64)
	grid_states = [np.array([i]) for i in var0]

	for S in grid_states:

		i_s = np.argmin(abs(var0-S[0]) )
		
		theta_inds = find_feature_inds(S,tilings,shift_tile_inds)

		A = np.max( np.sum(theta[theta_inds,:,:],axis=0), axis=1)

		Q[i_s,:] = A

	return Q




