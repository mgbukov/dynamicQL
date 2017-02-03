import numpy as np
from numpy import exp,fmod,searchsorted

from descrete_SE import get_H,V_mat
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
import matplotlib.pyplot as plt





def tweezers(x,x0,d0):
	return d0*exp(-32.0*(x-x0)**2) - 130.0*exp(-32.0*(x+x0)**2)


def tweezer(x,x0,d0):
	return d0*exp(-32.0*(x-x0)**2) 


def V(x,x0,d0):
	return V_mat(x,tweezers,(x0,d0))


def target_states(T,x,x_f,d_f):
	H_f = T + V_mat(x,tweezer,(+x_f,d_f))
	H_i = T + V_mat(x,tweezer,(-x_f,-130.0))

	[Ef],V_f = eigsh(H_f,k=1,which="SA",maxiter=10000)
	[Ef],V_i = eigsh(H_i,k=1,which="SA",maxiter=10000)

	return V_i.ravel(),V_f.ravel()



def Unitaries_1(T,x,dt,d,x_actions,x_range):
	dx = min(x_actions)
	x_min = min(x_range)
	x_max = max(x_range)

	expm_dict = {}

	n = int((x_max - x_min)/dx)

	for i in range(n+1):
		H = T + V_mat(x,tweezers,(i*dx,d))
		expm_dict[i] = expm(-1j*dt*H)

	return expm_dict