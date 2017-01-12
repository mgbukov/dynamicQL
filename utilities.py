'''
Created on Jan 2 , 2017

@author: Alexandre Day

'''
#from lib2to3.fixer_util import Number
import numpy as np

def check_sys_arg(argv):
	import sys
	"""
	Purpose:
    	Check command line arguments, (user might need help or has given wrong number of arguments)
	"""
	n_par=8
	message_1="""Expecting %s parameters from command line: 
	N_quench, N_time_step, action_set_name, outfile_name, max_fid_eval, delta_t, N_restart, verbose"""%n_par
	message_2=""" 
-- All parameters (including private) -- 

	L: system size
	J: Jzz interaction
	hz: longitudinal field
	hx_i: initial tranverse field coupling
	hx_initial_state: initial state transverse field
	hx_final_state: final state transverse field
	
	N_quench: number of quenches (i.e. no. of time temperature is quenched to reach exactly T=0)
	N_time_step: number of time steps
	action_set: array of possible actions
	outfile_name: file where data is being dumped (via pickle) 
	max_fid_eval: maximum number of fidelity evaluations
	delta_t: time scale
	N_restart: number of restart for the annealing
	verbose: If you want the program to print to screen the progress
	
	hx_max : maximum hx field (the annealer can go between -hx_max and hx_max
	FIX_NUMBER_FID_EVAL: decide wether you want to fix the maximum number of fidelity evaluations
	RL_CONSTRAINT: use reinforcement learning constraints or not
"""
	example="python LZ_sim_anneal.py 30 20 bang-bang8 out.txt 3000 0.05 100 False"
	
	
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
	
	_,N_quench,N_time_step,action_set_name,outfile_name,max_fid_eval,delta_t,N_restart,verbose=argv
	N_quench=int(N_quench)
	N_time_step=int(N_time_step)
	assert action_set_name in all_action_sets.keys(),"Wrong action set label, expecting one of the following: "+str(list(all_action_sets.keys()))
	action_set=all_action_sets[action_set_name]
	max_fid_eval=int(max_fid_eval)
	delta_t=float(delta_t)
	N_restart=int(N_restart)
	verbose=(verbose=="True")
	return N_quench,N_time_step,action_set,outfile_name,max_fid_eval,delta_t,N_restart,verbose,action_set_name

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

def make_file_name(params_SA):
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
					['N_time_step','int-3'],
					['N_quench','int-3'],
					['Ti','float-2'],
					['action_set','int-1'],
					['hx_initial_state','float-2'],
					['hx_final_state','float-2'],
					['delta_t','float-4'],
					['hx_i','float-2'],
					['RL_CONSTRAINT','bool'],
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
	
	file_name_composition=["SA","nStep-%s","nQuench-%s","Ti-%s","as-%s","hxIS-%s","hxFS-%s","deltaT-%s","hxI-%s","RL-%s","L-%s","J-%s","hz-%s"]
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
	
	for result,i in zip(result_all,range(N_sample)):
		count_fid_eval[i],best_fid[i],action_protocols[i],hx_protocols[i]=result    
	
	return count_fid_eval,best_fid,action_protocols,hx_protocols

def gather_data(params_SA,root):
	"""
	Purpose:
		Gather data produced by LZ_sim_anneal in nice format (dictionary, mapping param_values to the corresponding result)
	"""	
	import pickle
	
	file_name=root+make_file_name(params_SA)
	
	with open(file_name,'rb') as f:
		[_,result_all]=pickle.load(f)
	
	n_fid,fid,a_prot,h_prot=split_data(result_all,verbose=False)

	parsed_results={
				"n_fid":n_fid,
				"fid":fid,
				"action_protocol":a_prot,
				"h_protocol":h_prot,
				}
	
	return parsed_results
	
def check_version():
	import sys
	if sys.version_info[0] < 3:
		raise "Must be using Python 3"



