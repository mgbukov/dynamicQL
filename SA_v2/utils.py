'''
Created on Jan 2 , 2017

@author: Alexandre Day

'''
#from lib2to3.fixer_util import Number
import numpy as np
import os.path
import pickle

def f_to_str(number, prec=2): #this avoids rounding numbers and thus is more portable
        s="%.10f"%number 
        pos_p = s.find(".")
        s=s[:(pos_p+prec+1)]
        s=s.replace("-","m").replace(".","p")
        return s

def i_to_str(number,prec=2):
	l=len(str(number))
	if prec-l > 0:
		s="0"*(prec-l)+str(number)
	else:
		s=str(number)
	return s	

class UTILS:

	def __init__(self):
		self.param_type = {
		'L' : int,
		'dt' : float,
		'J' : float,
		'n_step': int,
		'hz' : float,
		'hx_i' : float,
		'hx_f' : float,
		'hx_max' : float,
		'hx_min' : float,
		'dh' : float,
		'n_sample' : int,
		'n_quench' : int,
		'Ti' : float,
		'T' : float,
		'symmetrize' : int,
		'outfile' : str,
		'verbose' : int,
		'task' : str,
		'root' : str,
		'fid_series': bool
	}

	def read_command_line_arg(self,parameters,argv):
		n_elem = len(argv)
		if n_elem > 1:
			for i, arg in zip(range(n_elem),argv):
				if i > 0:
					arg_split=arg.split('=')
					# if arg is not specified properly this will trigger a dictionnary key error <--
					parameters[arg_split[0]] = self.param_type[arg_split[0]](arg_split[1])
			
		if parameters['dt'] < 0. : # time slices should be automatically computed 
			parameters['dt'] = parameters['T']/parameters['n_step']
		else:
			parameters['T'] = parameters['dt']*parameters['n_step']

	def read_parameter_file(self, file="para.dat"):
		"""
		Reads parameters from file para.dat (default)
		Parameters are specified by a label and their value must be seperated by a space or a tab
		
		Return:
			dict of labels (str) to parameter values (float, str or int)
		"""
		with open(file,'r') as f:
			info={}
			for line in f:
				tmp=line.strip('\n').split('\t')
				if len(tmp) == 1:
					tmp = tmp[0].split(' ')
				assert len(tmp) == 2, 'Wrong format for input file, need to have a space or tab separating parameter and its value'
				info[tmp[0]]=tmp[1]
		param = {}
		for p in self.param_type.keys(): # cast strings to floats and ints !
			if self.param_type[p] == bool:
				if info[p] == 'True' or info[p] == '1':
					param[p] = True
				elif info[p] == 'False' or info[p] == '0':
					param[p] = False
				else:
					assert False, 'Wrong boolean format'
			else:
				param[p] = self.param_type[p](info[p])
		return param

	def print_parameters(self, parameters):

		""" 
		Parameters
			L: system size
			J: Jzz interaction
			hz: longitudinal field
			hx_i: initial tranverse field coupling
			hx_initial_state: initial state transverse field
			hx_final_state: final state transverse field
			Ti: initial temperature for annealing
			
			N_quench: number of quenches (i.e. no. of time temperature is quenched to reach exactly T=0)
			N_time_step: number of time steps
			action_set: array of possible actions
			outfile_name: file where data is being dumped (via pickle) 
			delta_t: time scale
			N_restart: number of restart for the annealing
			verbose: If you want the program to print to screen the progress
			symmetrize_protocol: Wether or not to work in the symmetrized sector of protocols
			
			hx_max : maximum hx field (the annealer can go between -hx_max and hx_max
			FIX_NUMBER_FID_EVAL: decide wether you want to fix the maximum number of fidelity evaluations (deprecated)
			RL_CONSTRAINT: use reinforcement learning constraints or not
			fidelity_fast: prepcompute exponential matrices and runs fast_Fidelity() instead of Fidelity()
		"""

		L, dt, J, n_step, hz, hx_max = tuple([parameters[s] for s in ['L','dt', 'J', 'n_step','hz','hx_max']])
		hx_initial_state, hx_final_state, n_quench, n_sample, n_step = tuple([parameters[s] for s in ['hx_i','hx_f','n_quench','n_sample','n_step']])
		symmetrize,outfile = tuple([parameters[s] for s in ['symmetrize','outfile']])

		print("-------------------- > Parameters < --------------------")
		print("{0:<30s}{1:<5d}".format('L',L))
		print("{0:<30s}{1:<5.2f}".format('J',J))
		print("{0:<30s}{1:<5.2f}".format('hz',hz))
		print("{0:<30s}{1:<5.2f}".format('hx_max',hx_max))
		print("{0:<30s}{1:<5.2f}".format('hx_initial_state',hx_initial_state))
		print("{0:<30s}{1:<5.2f}".format('hx_final_state',hx_final_state))

		print("{0:<30s}{1:<5d}".format('n_quench',n_quench))
		print("{0:<30s}{1:<5.2f}".format('dt',dt))
		print("{0:<30s}{1:<5d}".format('n_sample',n_sample))
		print("{0:<30s}{1:<5d}".format('n_step',n_step))
		
		print("{0:<30s}{1:<5.2f}".format('T',n_step*dt))
		print("{0:<30s}{1:<5s}".format('Task',parameters['task']))
		print("{0:<30s}{1:<5s}".format("Output file",'data/'+outfile))
		print("{0:<30s}{1:<5s}".format("Symmetrizing protocols",str(symmetrize)))
		print("{0:<30s}{1:<5s}".format("Compute fid series ?",str(parameters['fid_series'])))
		

	def make_file_name(self, parameters, extension = ".pkl", root=""):
		"""
		Purpose:
			Given the simulation parameters, produces a file name that contains the information about all the simulation parameters needed
			to reproduce the simulation results. Floats are converted to string where the dot "." is converted to "p" and the minus sign
			"-" is converted to "m"
		Return:
			String representing a file name (extension is always .pkl)
		"""
		if (len(root) != 0) and (root[-1] != "/"):
			root+="/"

		if parameters['outfile'] != 'auto':
			return root+parameters['outfile']
		
		# These parameters specify completely the simulation :
		param_and_type=[['task','str'],
						['L','int-2'],
						['dt','float-4'],
						['n_step','int-4'],
						['T','float-2'],
						# --
						['n_quench','int-4'],
						['Ti','float-2'],
						['symmetrize','int-1'],
						# -- 
						['J','float-2'],
						['hz','float-2'],
						['hx_i','float-2'],
						['hx_f','float-2'],
						['hx_max','float-2'],
						['hx_min','float-2'],
						['dh','float-2']
		]

		n_param=len(param_and_type)
		param_value=[0]*n_param
		
		for i,p in zip(range(n_param),param_and_type):
			param_name,cast_type=p
			tmp=cast_type.split('-')
			if tmp[0]=='float':
				param_value[i]=f_to_str(parameters[param_name],prec=int(tmp[1]))
			elif tmp[0] == 'int':
				param_value[i]=i_to_str(parameters[param_name],prec=int(tmp[1]))
			elif tmp[0] == 'str':
				param_value[i]=parameters[param_name]
			else:
				print(tmp)
				assert False,"Wrong cast-type format"
		
		file_name_composition=["%s",
								"L=%s","dt=%s","nStep=%s","T=%s",
								"nQuench=%s","Ti=%s","symm=%s",
								"J=%s","hz=%s","hxI=%s","hxF=%s","hxmax=%s","hxmin=%s","dh=%s"]
		file_name="_".join(file_name_composition)
		file_name=file_name%tuple(param_value)
		
		return root+file_name+extension

	def read_current_results(self, file_name):
		"""
			Read current data in filename
		"""
		if os.path.isfile(file_name):
			try:
				with open(file_name,'rb') as pkl_file:
					_ , all_results = pickle.load(pkl_file)
					pkl_file.close()
				n_sample = len(all_results)
				return n_sample, all_results
			except EOFError :
				return 0,[]
		else:
			return 0,[]

	def quick_setup(self,argv=[],file = 'para.dat'):

		argv_tmp=['myfile.txt']+argv
		from Hamiltonian import HAMILTONIAN
		from model import MODEL

		# Reading parameters from para.dat file
		parameters = self.read_parameter_file(file=file)

    	# Command line specified parameters overide parameter file values
		self.read_command_line_arg(parameters,argv_tmp)

		# Printing parameters for user
		print(parameters)
		#utils.print_parameters(parameters)

		# Defining Hamiltonian
		H = HAMILTONIAN(**parameters)

		# Defines the model, and precomputes evolution matrices given set of states
		return MODEL(H, parameters)


def parse_data(file, v=2):
	f=open(file,'rb')
	info, data = pickle.load(f)

	if v == 2:
		key = ['n_fid','F','E','n_visit','protocol']
	elif v == 3:
		key = ['n_fid','F','E','n_visit','protocol','fid_series']
	else:
		key = ['n_fid','F','E','protocol']

	n_sample = len(data)
	n_step = info['n_step']
	res = {}

	res['n_fid']=np.zeros(n_sample,dtype=np.int)
	res['n_visit']=np.zeros(n_sample,dtype=np.int)
	res['F']=np.zeros(n_sample)
	res['E']=np.zeros(n_sample)
	res['protocol'] = np.zeros((n_sample,n_step),dtype=np.int)
	res['fid_series'] = []

	for i, elem in enumerate(data):
		for j, ej in enumerate(elem):
			if key[j] == 'fid_series':
				res[key[j]].append(ej)
			else:
				res[key[j]][i] = ej
	return res

def check_version():
	import sys
	if sys.version_info[0] < 3:
		raise "Must be using Python 3"
