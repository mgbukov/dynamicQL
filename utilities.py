'''
Created on Jan 2 , 2017

@author: Alexandre Day

'''
from lib2to3.fixer_util import Number

def check_sys_arg(argv):
	import sys
	"""
	Purpose:
    	Check command line arguments, (user might need help or has given wrong number of arguments)
	"""
	n_par=8
	message="Expecting %s parameters from command line: n_quench, n_step, action_set_number,outfile_name,max_number_fid, dt, n_restart,verbose"%n_par
	example="python LZ_sim_anneal.py 30 20 bang-bang8 out.txt 3000 0.05 100 False"
	
	
	if len(argv)>1:
		if argv[1]=='-h':
			print(message)
			print("Example:\n\t"+example)
			exit()
		else:
			assert len(sys.argv) == n_par+1, message

def read_command_line_arg(argv,all_action_sets):
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

def make_file_name(params_SA):
	extension=".pkl"
	param_to_display=['N_time_step','N_quench','Ti','action_set','hx_initial_state','hx_final_state','delta_t','hx_i',  'RL_CONSTRAINT','L','J','hz']
	cast_type_spec=  ['int',        'int',     'float-2','int',      'float-2',        'float-2',       'float-2','float-2', 'bool','int','float-2','float-2']
	n_param=len(param_to_display)
	param_value=[0]*n_param
	
	for i,param,cast_type in zip(range(n_param),param_to_display,cast_type_spec):
		if (cast_type=='bool') or (cast_type=='int'):
			param_value[i]=str(int(params_SA[param]))
		else:
			tmp=cast_type.split('-')
			if tmp[0]=='float':
				param_value[i]=f_to_str(params_SA[param],prec=int(tmp[1]))
			else:
				print(tmp)
				assert False,"Wrong cast-type format"
	
	file_name_composition=["SA","nStep-%s","nQuench-%s","Ti-%s","as-%s","hxIS-%s","hxFS-%s","deltaT-%s","hxI-%s","RL-%s","L-%s","J-%s","hz-%s"]
	file_name="_".join(file_name_composition)+extension
	file_name=file_name%tuple(param_value)
	
	return file_name
		