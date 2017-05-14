'''
Created on Jan 2 , 2017

@author: Alexandre Day

'''
#from lib2to3.fixer_util import Number
import numpy as np
import os.path
import pickle

def check_sys_arg(argv):
	import sys
	"""
	Purpose:
    	Check command line arguments, (user might need help or has given wrong number of arguments)
	"""
	n_par=11
	message_1="""Expecting %s parameters from command line: 
	L, hxIS, hxFS, N_quench, N_time_step, action_set_name, outfile_name, delta_t, N_restart, verbose, symmetrize"""%n_par
	message_2=""" 
-- All parameters (including private) -- 

	J: Jzz interaction
	hz: longitudinal field
	hx_i: initial tranverse field coupling
	
	L: system size
	hx_initial_state: initial state transverse field
	hx_final_state: final state transverse field
	N_quench: number of quenches (i.e. no. of time temperature is quenched to reach exactly T=0)
	N_time_step: number of time steps
	action_set: array of possible actions
	outfile_name: file where data is being dumped (via pickle) 
	delta_t: time scale
	N_restart: number of restart for the annealing
	verbose: If you want the program to print to screen the progress
	
	hx_max : maximum hx field (the annealer can go between -hx_max and hx_max
	FIX_NUMBER_FID_EVAL: decide wether you want to fix the maximum number of fidelity evaluations
	RL_CONSTRAINT: use reinforcement learning constraints or not
"""
	example="python LZ_sim_anneal.py 8 -2. 2. 30 20 bang-bang8 out.txt 0.05 100 False False"
	
	
	if len(argv)>1:
		if argv[1]=='-h':
			print(message_1)
			print("Example:\n\t"+example+"\n")
			while True:
				mode=input('More info on parameters (y/n) ?')
				if mode == 'y':
					print(message_2)
					exit()
				elif mode == 'n':
					exit()
			
		else:
			assert len(sys.argv) == n_par+1, message_1

def read_command_line_arg(argv,all_action_sets):
	"""
	Purpose:
		Read command line argument that the user provides, casting strings to proper types (int and float).
	
	Parameters
	----------
	argv : array of strings, shape(n_parameters)
		Array of string of parsed command line
		
	all_action_set: dict, {string: array of floats, shape(n_action)}
		Dictionary map of action sets name ('bang-bang8,continuous,continuous-op') to corresponding
		array of possible actions 
	
	Returns:
		Tuple of parsed command line. Last element is the action set name.
	"""
	
	_, L, hxIS, hxFS, N_quench, N_time_step, action_set_name, outfile_name, delta_t, N_restart, verbose, symm = argv
	L = int(L)
	hxFS = float(hxFS)
	hxIS = float(hxIS)
	N_quench = int(N_quench)
	N_time_step = int(N_time_step)
	assert action_set_name in all_action_sets.keys(),"Wrong action set label, expecting one of the following: "+str(list(all_action_sets.keys()))
	action_set = all_action_sets[action_set_name]
	delta_t = float(delta_t)
	N_restart = int(N_restart)
	verbose = (verbose == "True")
	symm = (symm == "True")
	return L, hxIS, hxFS, N_quench, N_time_step, action_set, outfile_name, delta_t, N_restart, verbose, action_set_name, symm

def read_parameter_file(file="para.dat"):
	with open(file,'r') as f:
		info={}
		for line in f:
			tmp=line.strip('\n').split('\t')
			info[tmp[0]]=tmp[1]

	param_type = {
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
		'symmetrize' : int,
		'outfile' : str
	}

	param = {}
	for p in param_type.keys(): # cast strings to floats and ints !
		param[p] = param_type[p](info[p])

	return param

def print_parameters(parameters):

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

	L, dt, J, n_step, hz, hx_i, hx_max = tuple([parameters[s] for s in ['L','dt', 'J', 'n_step','hz','hx_i','hx_max']])
	hx_initial_state, hx_final_state, n_quench, n_sample, n_step = tuple([parameters[s] for s in ['hx_i','hx_f','n_quench','n_sample','n_step']])
	symmetrize,outfile = tuple([parameters[s] for s in ['symmetrize','outfile']])

	print("-------------------- > Parameters < --------------------")
	print("L \t\t\t %i\nJ \t\t\t %.3f\nhz \t\t\t %.3f\nhx(t=0) \t\t %.3f\nhx_max \t\t\t %.3f "%(L, J, hz, hx_i, hx_max))
	print("hx_initial_state \t %.2f\nhx_final_state \t\t %.2f"%(hx_initial_state, hx_final_state))
	print("N_quench \t\t %i\ndelta_t \t\t %.2f\nN_restart \t\t %i"%(n_quench, dt, n_sample))
	print("N_time_step \t\t %i"%n_step)
	print("Total_time \t\t %.2f"%(n_step*dt))
	print("Output file \t\t %s"%('data/'+outfile))
	#print("# of possible actions \t %i"%len(action_set))
	#print("Action_set \t <- \t %s"%action_set)
	print("Symmetrizing protocols \t %s"%str(symmetrize))
	#print("Fidelity MODE \t\t %s"%('fast' if fidelity_fast else 'standard'))

def f_to_str(number,prec=2):
	s=("%."+str(prec)+"f")%number
	s=s.replace("-","m").replace(".","p")
	return s	

def i_to_str(number,prec=2):
	l=len(str(number))
	if prec-l > 0:
		s="0"*(prec-l)+str(number)
	else:
		s=str(number)
	return s	

def make_file_name(parameters, extension = ".pkl", root=""):
	"""
	Purpose:
		Given the simulation parameters, produces a file name that contains the information about all the simulation parameters needed
		to reproduce the simulation results. Floats are converted to string where the dot "." is converted to "p" and the minus sign
		"-" is converted to "m"
	Return:
		String representing a file name (extension is always .pkl)
	"""
	if parameters['outfile'] != 'auto':
		return root+parameters['outfile']
	
	# These parameters specify completly the simulation :
	param_and_type=[['L','int-2'],
					['dt','float-4'],
					['n_step','int-4'],
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
					['dh','float-2'],
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
		else:
			print(tmp)
			assert False,"Wrong cast-type format"
	
	file_name_composition=["L=%s","dt=%s","nStep=%s",
							"nQuench=%s","Ti=%s","symm=%s",
							"J=%s","hz=%s","hxI=%s","hxF=%s","hxmax=%s","hxmin=%s","dh=%s"]
	file_name="_".join(file_name_composition)
	file_name=file_name%tuple(param_value)
	
	return root+file_name+extension

def make_unitary_file_name(params_SA):
	"""
	Purpose:
		Given the simulation parameters, produces a file name that contains the information about all the simulation parameters needed
		to reproduce the simulation results. Floats are converted to string where the dot "." is converted to "p" and the minus sign
		"-" is converted to "m"
	Return:
		String representing a file name (extension is always .pkl)
	"""
	
	extension=".pkl"
	param_and_type=[
					['action_set','int-1'],
					['delta_t','float-4'],
					['hx_i','float-2'],
					['L','int-2'],
					['J','float-2'],
					['hz','float-2']
	]
	n_param=len(param_and_type)
	param_value=[0]*n_param
	
	for i,p in zip(range(n_param),param_and_type):
		param_name,cast_type=p
		if cast_type=='bool':
			param_value[i]=str(int(params_SA[param_name]))
		else:
			tmp=cast_type.split('-')
			if tmp[0]=='float':
				param_value[i]=f_to_str(params_SA[param_name],prec=int(tmp[1]))
			elif tmp[0] == 'int':
				param_value[i]=i_to_str(params_SA[param_name],prec=int(tmp[1]))
			else:
				print(tmp)
				assert False,"Wrong cast-type format"
	
	file_name_composition=["SA","as-%s","deltaT-%s","hxI-%s","L-%s","J-%s","hz-%s"]
	file_name="_".join(file_name_composition)+extension
	file_name=file_name%tuple(param_value)
	
	return file_name

def split_data(result_all,verbose=True):
	N_time_step=len(result_all[0][2])
	N_sample=len(result_all)
	if verbose:
		print("--- > N_sample=%i,\t N_time_step=%i"%(N_sample,N_time_step))
	action_protocols=np.empty((N_sample,N_time_step),dtype=np.float32)
	hx_protocols=np.empty((N_sample,N_time_step),dtype=np.float32)
	count_fid_eval=np.empty((N_sample,),dtype=np.int32)
	best_fid=np.empty((N_sample,),dtype=np.float32)
	E=np.empty((N_sample,),dtype=np.float32)
	delta_E=np.empty((N_sample,),dtype=np.float32)
	Sd=np.empty((N_sample,),dtype=np.float32)
	Sent=np.empty((N_sample,),dtype=np.float32)

	for result,i in zip(result_all,range(N_sample)):
		count_fid_eval[i],best_fid[i],action_protocols[i],hx_protocols[i], E[i], delta_E[i], Sd[i], Sent[i] = result 

	return count_fid_eval,best_fid,action_protocols,hx_protocols, E, delta_E, Sd, Sent

def gather_data(params_SA,root):
	"""
	Purpose:
		Gather data produced by LZ_sim_anneal in nice format (dictionary, mapping param_values to the corresponding result)
	"""	
	
	file_name=root+make_file_name(params_SA)
	
	if os.path.isfile(file_name): 
		with open(file_name,'rb') as f:
			[_,result_all]=pickle.load(f)
	
		n_fid, fid, a_prot, h_prot, E, delta_E, Sd, Sent = split_data(result_all,verbose=False)
	
		parsed_results={
					"n_fid" : n_fid,
					"fid" : fid,
					"action_protocol" : a_prot,
					"h_protocol" : h_prot,
					"E" : E,
					"delta_E" : delta_E,
					"Sd" : Sd,
					"Sent" : Sent
		}
		return parsed_results
		
def check_version():
	import sys
	if sys.version_info[0] < 3:
		raise "Must be using Python 3"

def default_parameters():
	"""
	Returns default paramters
	"""
	param={'N_time_step':10,
		   'N_quench':0,
		   'Ti':0.04,
		   'action_set':0,
		   'hx_initial_state':-2.0,
		   'hx_final_state':2.0,
			'delta_t':0.01,
			'hx_i':-4.0,
			'RL_CONSTRAINT':True,
			'L':1,
			'J':1.00,
			'hz':1.0,
			'symmetrize':True
	}
	return param
			

def read_current_results(file_name):
	"""
		Read current data in filename
	"""
	if os.path.isfile(file_name):
		with open(file_name,'rb') as pkl_file:
			_ , all_results = pickle.load(pkl_file)
			pkl_file.close()
		n_sample = len(all_results)
		return n_sample, all_results
	else:
		return 0, []