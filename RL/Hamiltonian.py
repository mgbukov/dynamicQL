from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.operators import exp_op

import numpy as np

import time
import sys
import os
import cPickle



def Hamiltonian(L,J,hz,fun=None,fun_args=[]):
	######## define physics
	basis = spin_basis_1d(L=L,kblock=0,pblock=1,pauli=False) #
			
	zz_int =[[J,i,(i+1)%L] for i in range(L)]
	if L==1:
		x_field=[[1.0,i] for i in range(L)]
		z_field=[[hz,i] for i in range(L)]
	else:
		x_field=[[-1.0,i] for i in range(L)]
		z_field=[[-hz,i] for i in range(L)]

	static = [["zz",zz_int],["z",z_field]]
	dynamic = [["x",x_field,fun,fun_args]]

	kwargs = {'dtype':np.float64,'basis':basis,'check_symm':False,'check_herm':False,'check_pcon':False}
	H = hamiltonian(static,dynamic,**kwargs)

	return H


def Unitaries(delta_time,L,J,hz,action_min,var_max,var_min,state_i,save=False,save_str=''):

	# define Hamiltonian
	b=0.0
	lin_fun = lambda t: b 
	H = Hamiltonian(L,fun=lin_fun,**{'J':J,'hz':hz})

	# number of unitaries
	n = int((var_max - var_min)/action_min)
	
	# preallocate dict
	expm_dict = {}
	for i in range(n+1):
		# define matrix exponential; will be changed every time b is overwritten
		b = state_i[0]+i*action_min
		expm_dict[i] = np.asarray( exp_op(H,a=-1j*delta_time).get_mat().todense() )
		
	if save:

		### define save directory for data
		# read in local directory path
		str1=os.getcwd()
		str2=str1.split('\\')
		n=len(str2)
		my_dir = str2[n-1]
		# create directory if non-existant
		save_dir = my_dir+"/unitaries"
		if not os.path.exists(save_dir):
		    os.makedirs(save_dir)
		save_dir="unitaries/"

		# save file
		dataname  = save_dir + "unitaries"+save_str+'.pkl'
		cPickle.dump(expm_dict, open(dataname, "wb" ) )

	else:
		return expm_dict

