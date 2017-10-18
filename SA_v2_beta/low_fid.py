import pickle
import numpy as np
from utils import UTILS

def main():

	### READING ES data and finding optimal state and the gap to excited states.
	### Looping over parameters and storing data in a dictionary. 
	
	utils=UTILS()
	parameters = utils.read_parameter_file()

	Ts=np.arange(1.5,2.81,0.05)
	n_step=28
	

	fidelities=np.zeros((n_step,201),np.float64)
	energies=np.zeros_like(fidelities)

	protocols=np.zeros_like(fidelities)

	for i_,T in enumerate(Ts):
		
		parameters['T']= T
		parameters['n_step'] = n_step
		parameters['dt'] = T/n_step

		b2_array = lambda n10 : np.array(list(np.binary_repr(n10, width=n_step)), dtype=np.int)
		

		file = utils.make_file_name(parameters,root='data/data_ES/')
		res = {}
		with open(file,'rb') as f:
			data = pickle.load(f)

		h_index=np.argsort(-data[:,0])[:201]

		fidelities[i_,:]=data[h_index,0]
		energies[i_,:]=data[h_index,1]

		protocols[i_,:]=h_index

		print("done with %i/%i iterations." %(i_+1,len(Ts)))


	save_name='processed_data/low-fid_spectrum_L=2_J=1_hz=1.txt'
	pickle.dump([fidelities,energies,protocols], open(save_name, "wb" ) )














if __name__ == "__main__":
	main()