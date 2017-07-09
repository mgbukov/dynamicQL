'''
Created on Sep 1 , 2016

@author: Alexandre Day
'''
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import numpy as np

class HAMILTONIAN:
	# Time dependent Hamiltonian class
	def __init__(self, L=1 , J=1.0, hz = 1.0, hx_min=-4., hx_max =4., dh=2., n_step=1, **kwargs): # N is the number of time steps
		#basis = spin_basis_1d(L=L,pauli=False)
		fct_arg=[]
		ones=[[-1,i] for i in range(L)]
		z_field=[[-hz,i] for i in range(L)]

		if L>1:
			basis = spin_basis_1d(L=L,pauli=False,kblock=0,pblock=1) # include symmetries (momentum and parity sectors)
			zz_int =[[-J,i,(i+1)%L] for i in range(L)] # Has periodic boundary conditions
			static = [["zz",zz_int], ["z",z_field]]
		else:
			basis = spin_basis_1d(L=L,pauli=False) # w/o symmetries (momentum and parity sectors)
			static = [["z", z_field]]
		

		self.h_set = self.compute_h_set(hx_min,hx_max,dh) # discrete set of possible h fields 
		self.hx_discrete = np.zeros(n_step, dtype=int) # hx_discrete are protocols specified as integers

		fct = lambda time: self.h_set[self.hx_discrete[int(time)]] # time takes discrete values in our problem
		fct_cont = lambda h: h # trick : when calling the time - will instead interpret it as a field value 

		dynamic_discrete = [["x", ones, fct, fct_arg]]
		dynamic_cont = [["x", ones, fct_cont, fct_arg]]
		
		kwargs = {'dtype':np.float64,'basis':basis,'check_symm':False,'check_herm':False,'check_pcon':False}

		self.basis = basis
		self.hamiltonian_discrete = hamiltonian(static, dynamic_discrete, **kwargs)
		self.hamiltonian_cont = hamiltonian(static, dynamic_cont, **kwargs)

	def ground_state(self, hx = 0.):
		return self.hamiltonian_cont.eigsh(time=hx, k=1, which='SA')[1]

	def eigen_basis(self, hx = 0.):
		return self.hamiltonian_cont.eigh(time=hx)

	def compute_h_set(self, hmin, hmax, dh):
		return np.arange(hmin, hmax+1e-6, dh)

	def update_hx(self, time=0, hx = None, hx_idx = None):
		if hx_idx is not None:
			self.hx_discrete[time]=hx_idx
		elif hx is not None:
			idx = np.argmin(np.abs(self.h_set-hx))
			self.hx_discrete[time] = idx
		else:
			assert False, "Error in update_hx in Hamiltonian class"

	def evaluate_H_at_hx(self, hx = 0.):
		return self.hamiltonian_cont(time = hx)