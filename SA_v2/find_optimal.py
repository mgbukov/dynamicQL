import pickle
import numpy as np
from utils import UTILS

def main():

    ### READING ES data and finding optimal state and the gap to excited states.
    ### Looping over parameters and storing data in a dictionary. 
    
    utils=UTILS()
    parameters = utils.read_parameter_file()
    res = {} 
    for T in np.arange(0.1,4.01,0.1):
        for n_step in [4,6,8,10,12,14,16,18,20,22,24]:
            parameters['T']= T
            parameters['n_step'] = n_step
            parameters['dt'] = T/n_step
            
            file = utils.make_file_name(parameters,root='data/')
            with open(file,'rb') as f:
                data = pickle.load(f)
            
            pos_min = np.argmax(data[:,0])
            optimal_fid = data[pos_min,0]
            data[pos_min,0]-=10.
            pos_min_2 = np.argmax(data[:,0])
            gap = optimal_fid - data[pos_min_2,0]
            print(T,'\t',n_step,'\t',optimal_fid,'\t',gap)
            res[(round(T,2),n_step)]=[optimal_fid, gap]
    with open('optimal.pkl','wb') as f:
        pickle.dump(res,f)















if __name__ == "__main__":
    main()
