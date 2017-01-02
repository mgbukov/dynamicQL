'''
Created on Jan 2 , 2017

@author: Alexandre Day

'''

def check_sys_arg(argv):
	import sys
	"""
	Purpose:
    	Check command line arguments, (user might need help or has given wrong number of arguments)
	"""
	message="Expecting 7 parameters from command line: n_quench, n_step, action_set_number,outfile_name, max_number_fid, dt, n_restart"
	example="python LZ_sim_anneal.py 30 20 bang-bang8 out.txt 3000 0.05 100"
	
	if len(argv)>1:
		if argv[1]=='-h':
			print(message)
			print("Example:\n\t"+example)
			exit()
		else:
			assert len(sys.argv) == 8, message

def read_command_line_arg(argv,all_action_sets):
	_,N_quench,N_time_step,action_set,outfile_name,max_fid_eval,delta_t,N_restart=argv
	N_quench=int(N_quench)
	N_time_step=int(N_time_step)
	assert action_set in all_action_sets.keys(),"Wrong action set label, expecting one of the following: "+str(list(all_action_sets.keys()))
	action_set=all_action_sets[action_set]
	max_fid_eval=int(max_fid_eval)
	delta_t=float(delta_t)
	N_restart=int(N_restart)
	return N_quench,N_time_step,action_set,outfile_name,max_fid_eval,delta_t,N_restart
	
		