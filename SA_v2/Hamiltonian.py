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
			basis = spin_basis_1d(L=L,pauli=False)#,kblock=0,pblock=1) # include symmetries (momentum and parity sectors)
			zz_int =[[-J,i,(i+1)%L] for i in range(L)] # Has periodic boundary conditions
			static = [["zz",zz_int], ["z",z_field]]
		else:
			basis = spin_basis_1d(L=L,pauli=False) # w/o symmetries (momentum and parity sectors)
			static = [["z", z_field]]
		

		self.h_set = self.compute_h_set(hx_min,hx_max,dh) # discrete set of possible h fields 
		self.hx_discrete = [0]*n_step # hx_discrete are protocols specified as integers
		fct = lambda time: self.h_set[self.hx_discrete[int(time)]]
		dynamic = [["x", ones, fct, fct_arg]]
		
		kwargs = {'dtype':np.float64,'basis':basis,'check_symm':False,'check_herm':False,'check_pcon':False}

		self.model = hamiltonian(static, dynamic, **kwargs)
		self.basis = basis
		self.hamiltonian = hamiltonian(static, dynamic, **kwargs)
		self.basis = basis

	def ground_state(self,time=0,hx=0):
		return self.hamiltonian.eigsh(time=time,k=1,which='SA')[1]

	def eigen_basis(self,time=0):
		return self.hamiltonian.eigsh(time=time,which='SA')[1]
	
	def compute_h_set(self,hmin,hmax,dh):
		return np.arange(hmin,hmax+1e-6,dh)
	
	def update_hx_real(self,time=0,hx=0):
		idx = np.argmin(np.abs(self.h_set-hx))
		self.hx_discrete[time] = idx

	def update_hx(self,time,hx_idx):
		self.hx_discrete[time]=hx_idx

	def evaluate_ground_state_hx_special(self,hx=0):
		t1,t2 = self.hx_discrete[0], self.h_set[0]
		self.hx_discrete[0]=0
		self.h_set[0]=hx
		gs = self.hamiltonian.eigsh(time=0,k=1,which='SA')[1]
		self.hx_discrete[0], self.h_set[0]= t1,t2
		return gs
		
	def evaluate_H_hx_special(self,hx=0):
		t1,t2 = self.hx_discrete[0], self.h_set[0]
		self.hx_discrete[0]=0
		self.h_set[0]=hx
		H = self.hamiltonian.toarray().copy()
		self.hx_discrete[0], self.h_set[0]= t1,t2
		return H
		


	######### ---- > left it at trying to figure out how to specify an arbitrary field ... 
	############_                 