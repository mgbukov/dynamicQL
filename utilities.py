def check_sys_arg(argv):
	import sys
	"""
	Purpose:
    	Check command line arguments, (user might need help or has given wrong number of arguments)
	"""
	if len(argv)>1:
		if argv[1]=='-h':
			print("Expecting 5 parameters from command line: n_step, action_set_number,outfile_name, max_number_fid, dt")
			print("Example:\n\tpython LZ_sim_anneal.py 20 1 out.txt 3000 0.05")
			exit()
		else:
			assert len(sys.argv) == 6, "Wrong number of parameters, expecting 5 parameters : n_step, action_set_number,outfile_name, max_number_fid, dt"