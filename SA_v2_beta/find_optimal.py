import pickle
import numpy as np
from utils import UTILS

def main():

    ### READING ES data and finding optimal state and the gap to excited states.
    ### Looping over parameters and storing data in a dictionary. 
    
    utils=UTILS()
    parameters = utils.read_parameter_file()

    
    for T in np.arange(1.5,2.81,0.05): #[1.8,2.15]:#[0.6,0.8,1.2,1.4,1.5,1.6,1.7,1.8,2.0,2.1,2.15,2.2,2.4,2.6,2.8]:
        for n_step in [28]:
            
            parameters['T']= T
            parameters['n_step'] = n_step
            parameters['dt'] = T/n_step

            b2_array = lambda n10 : np.array(list(np.binary_repr(n10, width=n_step)), dtype=np.int)
            
            file = utils.make_file_name(parameters,root='data/data_ES/')
            res = {}
            with open(file,'rb') as f:
                data = pickle.load(f)

            
            pos_min = np.argmax(data[:,0])
            optimal_fid = data[pos_min,0]
            data[pos_min,0]-=10.
            pos_min_2 = np.argmax(data[:,0])
            gap = optimal_fid - data[pos_min_2,0]
            print("T = %0.3f: F_optimal = %0.5f; gap = %0.10f" %(T, optimal_fid, gap) )
            res[(T,n_step)]=[optimal_fid, gap]
            #exit()

            print(pos_min)
            optimal_h=b2_array(pos_min)
            optimal_h[np.abs(optimal_h)<1E-12]=-1

            print(optimal_h)
            print(optimal_h + optimal_h[::-1])
            print()
            #exit()  














if __name__ == "__main__":
    main()