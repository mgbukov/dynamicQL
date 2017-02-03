from descrete_SE import laplace_mat,V_mat
from tweezer_hamiltonian import tweezers
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
import numpy as np
import cProfile as cp






L=400
dt = 1.0/200
x = np.linspace(-1,1,L)
psi = np.zeros(L,dtype=np.float64)
T = laplace_mat(x,x[1]-x[0],periodic=True)
H = T + V_mat(x,tweezers,(-0.5,-160))

expH = expm(-1j*dt*H)

cp.run("expm_multiply((-1j*H*dt),psi)")
cp.run("expH.dot(psi)")