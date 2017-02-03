from scipy.sparse import dia_matrix
from numpy import ones_like,array,vstack,float32


# constructs a diagonal matrix with the function as the diagonal
def V_mat(x,V_func,V_args,dtype=float32):
	n = x.shape[0]
	ff = V_func(x,*V_args)
	return  dia_matrix((ff,[0]),shape=(n,n),dtype=dtype)


# calculates the descrete version of the laplacian operator
def laplace_mat(x,dx,dtype=float32,periodic=False):
	n = x.shape[0]
	od = ones_like(x)/dx**2
	d = -2*array(od)

	data = vstack((od,d,od))
	mat = dia_matrix((data,[-1,0,1]),shape=(n,n),dtype=dtype)

	if periodic:
		mat = mat.tocsr()
		mat[0,n-1] = 1/dx**2
		mat[n-1,0] = 1/dx**2
		mat = mat.todia()

	return mat


def get_H(x,dx,m,V_func,V_args,periodic=False,dtype=float32):
	T = (-1.0/(2*m))*laplace_mat(x,dx,periodic=periodic,dtype=dtype)
	V = V_mat(x,V_func,V_args,dtype=dtype)
	return T + V

