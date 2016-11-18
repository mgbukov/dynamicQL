import numpy as np
import scipy.linalg as _la
import scipy.sparse.linalg as _sla
import numpy.random as random
import Reinforcement_Learning as RL

random.seed(0)

max_t_steps = 20 #40

N_tilings = 20 #20
N_lintiles = 20
N_tiles = N_lintiles
N_actions = RL.actions_length()

N_episodes = 2000
# discount rate
gamma = 1.0
# learning rate
alpha_0 = 0.9#/N_tilings
# usage eta
eta = 0.6#/N_tilings
# TD(lambda) parameter
lmbda = 1.0

u0 = 1.0/alpha_0*np.ones((N_tiles*N_tilings,max_t_steps,N_actions), dtype=np.float64)
#u0 = 1.0/alpha_0*np.ones((N_tiles,), dtype=np.float64)
v0 = np.zeros(u0.shape, dtype=np.float64)


for j in xrange(N_episodes):
		# set initial usage vector
		u = u0.copy()
		# set initial aux v vector
		v = v0.copy()

		#print u.shape

		indA=0
		theta_inds = [0]

		# generate episode
		for t_step in xrange(max_t_steps):

			rho = 2.0 #random.uniform(0,1)


			with np.errstate(divide='ignore'):
				alpha = 1.0/(u[theta_inds,t_step,indA]*N_tilings)
				alpha[u[theta_inds,t_step,indA]<1E-12] = 1.0
				#alpha = 1.0/(u[theta_inds]*N_tilings)
				#alpha[u[theta_inds]<1E-12] = 1.0


			print 'rho*alpha, rho*alpha_feat rho, cutoff', max(alpha), rho*max(alpha), rho, 1.0/N_tilings
			print '_____'

			#exit()

			#print "ALPHA", rho*alpha_0/N_tilings, rho*max(alpha), rho*min(alpha)
			#if max(alpha) > 1.0/N_tilings:
			"""
			if rho*max(alpha) > 1.0/N_tilings:
				exit()
			"""
			if t_step == max_t_steps-1:
				break	

			#"""
			u[:,t_step+1,:] = u[:,t_step,:] + (rho-1.0)*gamma*lmbda*v[:,t_step,:]
			u[theta_inds,t_step+1,indA] += -eta*u[theta_inds,t_step,indA] + rho - (rho-1.0)*gamma*lmbda*eta*v[theta_inds,t_step,indA]
			# update aux v vector
			v[:,t_step+1,:] = gamma*lmbda*rho*v[:,t_step,:]
			v[theta_inds,t_step+1,indA] += rho - eta*gamma*rho*lmbda*v[theta_inds,t_step,indA]
			#"""

			"""
			# calculate usage and alpha vectors
			u[theta_inds] *= (1.0-eta)
			u += (rho-1.0)*gamma*lmbda*v
			u[theta_inds] += rho - (rho-1.0)*gamma*lmbda*eta*v[theta_inds]
			# update aux v vector
			v *= gamma*lmbda*rho
			v[theta_inds] *= 1.0-eta
			v[theta_inds] += rho
			"""

		exit()


