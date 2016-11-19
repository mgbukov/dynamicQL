from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d

import numpy as np



def Hamiltonian(L,J,hz,fun=None,fun_args=[]):
	######## define physics
	basis = spin_basis_1d(L=L,kblock=0,pblock=1,pauli=False)
			
	zz_int =[[J,i,(i+1)%L] for i in range(L)]
	if L==1:
		x_field=[[1.0,i] for i in range(L)]
		z_field=[[hz,i] for i in range(L)]
	else:
		x_field=[[-1.0,i] for i in range(L)]
		z_field=[[-hz,i] for i in range(L)]

	static = [["zz",zz_int],["z",z_field]]
	dynamic = [["x",x_field,fun,fun_args]]

	kwargs = {'dtype':np.float64,'basis':basis,'check_symm':False,'check_herm':False}
	H = hamiltonian(static,dynamic,**kwargs)

	return H

