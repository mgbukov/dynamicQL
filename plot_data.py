from Bloch import *
import numpy as np

import Hamiltonian

from quspin.tools.measurements import ent_entropy
from quspin.operators import exp_op

from pylab import *
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import os, sys

#read in local directory path
str1=os.getcwd()
str2=str1.split('\\')
n=len(str2)
my_dir = str2[n-1]

#####


def plot_rewards(Fidelity_ep,Return,Return_ave,save_name,data_params,save=True):
	""" This function plots the rewards vs episodes. """

	str_R = "$\\mathrm{episodic}$"
	str_Rave = "$\\mathrm{average}$"
	str_F = "$\\mathrm{fidelity}$"

	N_episodes = len(Fidelity_ep)
	
	plt.plot(xrange(N_episodes),Fidelity_ep,'g',linewidth=2.0,label=str_F)
	plt.plot(xrange(N_episodes),Return,'r-.',linewidth=0.5,label=str_R)
	plt.plot(xrange(N_episodes),Return_ave,'b',linewidth=2.0,label=str_Rave)
	

	#plt.xscale('log')

	plt.xlabel('$\\#\\ \\mathrm{episodes}$', fontsize=20)
	plt.ylabel('$\\mathrm{reward}$', fontsize=20)

	plt.legend(loc='lower right')
	plt.tick_params(labelsize=16)
	plt.grid(True)

	save_dir = my_dir+"/data"
	if not os.path.exists(save_dir):
	    os.makedirs(save_dir)

	if save:
		save_str = "data/"+save_name+data_params+'.png'
		plt.savefig(save_str)

	#plot.show()
	plt.close()


def plot_protocols(times,protocol,observable,save_name,data_params,save=False):
	""" This function plots the protocols + observable vs. time."""

	# extend time and protocol vecors
	delta_times=times[1]-times[0]
	times=np.append(times,times[-1]+delta_times)
	protocol=np.append(protocol,protocol[-1])


	str_p="$h_x(t)$"
	if save_name=='fid':
		str_f="$F(t)$"
	elif save_name=='en':
		str_f="$E(t)-E_\\mathrm{gs}(t)$"
	elif save_name=='en_fluct':
		str_f="$\delta E(t)$"
	elif save_name=='sd':
		str_f="$S_\\mathrm{d}^\\mathrm{target}$"
	elif save_name=='sent':
		str_f="$S_\\mathrm{ent}(t)$"
	
	
	plt.step(times,protocol,'b',marker='.',where='post',linewidth=1,label=str_p)
	plt.plot(times,observable,'b--',linewidth=1,label=str_f)

	t_max = times[-1]
	p_max = max( max(protocol),max(observable) ) 
	p_min = min( min(protocol),min(observable) )


	plt.xlabel('$t$', fontsize=20)
	#plt.ylabel('$h_x(t)$', fontsize=20)

	#plt.xlim([0,t_max])
	plt.ylim([p_min-0.5,p_max+0.5])

	plt.legend(loc='best')
	plt.tick_params(labelsize=16)
	plt.grid(True)

	save_dir = my_dir+"/data"
	if not os.path.exists(save_dir):
	    os.makedirs(save_dir)

	if save:
		save_str = "data/"+save_name+data_params+'.png'
		plt.savefig(save_str)

	#show()
	plt.close()



def observables(L,times,protocol,hx_i,hx_f,J,hz,data_params,save=True):

	b=hx_f	
	lin_fun = lambda t: b
	# define Hamiltonian
	H = Hamiltonian.Hamiltonian(L,fun=lin_fun,**{'J':J,'hz':hz})
	# define matrix exponential
	delta_times=times[1]-times[0]
	exp_H=exp_op(H,a=-1j*delta_times)

	# calculate final basis
	_,V_target=H.eigh()

	# calculate initial state
	b=hx_i
	E_i,psi=H.eigsh(k=1,which='SA',maxiter=1E10,return_eigenvectors=True)
	#E_i, psi=H.eigsh(k=1,sigma=-0.1,maxiter=1E10,return_eigenvectors=True)
	psi=psi.squeeze()

	subsys=[i for i in range(L/2)]

	# preallocate variables
	Fidelity,E,delta_E,Sd,Sent=[],[],[],[],[]
	# create Bloch clas object
	points=[]
	i=0
	while True:
		# instantaneous fidelity
		Fidelity.append( abs(psi.conj().dot(V_target[:,0]))**2 )
		# excess energy above inst energy
		EGS = H.eigsh(k=1,which='SA',maxiter=1E10,return_eigenvectors=False)
		E.append( H.matrix_ele(psi,psi).real/L - EGS/L)
		# inst energy density
		delta_E.append( np.sqrt( (H(time=0)*H).matrix_ele(psi,psi).real - H.matrix_ele(psi,psi).real**2 + 1E2*np.finfo(psi[0].dtype).eps )/L   )
		# diagonal entropy in target basis
		pn = abs( V_target.conj().T.dot(psi) )**2.0 + np.finfo(psi[0].dtype).eps
		Sd.append( -pn.dot(np.log(pn))/L )
		# entanglement entropy
		if L!=1:
			Sent.append( ent_entropy(psi,H.basis,chain_subsys=subsys)['Sent'] )
		else:
			# Bloch sphere image
			bloch = Bloch()
			
			# plot the states as arrow
			bloch.add_states([psi,V_target[:,0]])
			# extract spherical coordinates of psi
			points.append(bloch.vectors[0])
			# plot all psi's as blue dots
			bloch.add_points([list(a) for a in zip(*points)],meth='l')
			# plot psi_target
			bloch.add_points(bloch.vectors[1])
			# set view angle
			bloch.view = [40,10]
			bloch.set_label_convention("original")
			#bloch.show()
			#create temporary save directory
			save_dir = my_dir+"/temp"
			if not os.path.exists(save_dir):
			    os.makedirs(save_dir)
			bloch.save(name='temp/bloch_{}.png'.format(i))
		

		if i == len(protocol):
			break
		else:
			# go to next step
			b=protocol[i] # --> induces a change in H
			psi = exp_H.dot(psi)
			i+=1
			

	#create temporary save directory
	save_dir = my_dir+"/data"
	if not os.path.exists(save_dir):
	    os.makedirs(save_dir)

	
	# plot protocol
	plot_protocols(times,protocol,Fidelity,'fid',data_params,save=save)
	plot_protocols(times,protocol,E,'en',data_params,save=save)
	plot_protocols(times,protocol,delta_E,'en_fluct',data_params,save=save)
	plot_protocols(times,protocol,Sd,'sd',data_params,save=save)
	if L!=1:
		plot_protocols(times,protocol,Sent,'sent',data_params,save=save)
	else:
		# create movie
		movie_name="data/blochsphere_" + data_params
		cmd = "ffmpeg -framerate 5 -i temp/bloch_%01d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+movie_name+".mp4"
		# execute command cmd
		os.system(cmd)
		# remove temp directory
		os.system("rm -rf temp*")


