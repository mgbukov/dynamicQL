import pickle
import numpy as np
from utils import UTILS



def gamma(hx_tmp):
    n_step = len(hx_tmp)
    prod = hx_tmp[:n_step // 2]*(hx_tmp[n_step // 2:][::-1])
    pup = len(prod[prod > 0])
    return (1.*pup) / n_step

def b2_array(n10,n) :
    return np.array(list(np.binary_repr(n10, width=n)), dtype=np.float)


np.set_printoptions(suppress=True)
def main():

    ### READING ES data and finding optimal state and the gap to excited states.
    ### Looping over parameters and storing data in a dictionary. 
    
    utils=UTILS()
    parameters = utils.read_parameter_file()
    res = {}
    
    for T in [3.6]:
        for n_step in [24]:
            parameters['T']= T
            parameters['n_step'] = n_step
            parameters['dt'] = T/n_step
            
            file = utils.make_file_name(parameters,root='data/')
            with open(file,'rb') as f:
                data = pickle.load(f)
            
            print(np.sort(data[:,0])[-8:])
            exit()
            pos_min = np.argmax(data[:,0])
            optimal_fid = data[pos_min,0]
            data[pos_min,0]-=10.
            pos_min_2 = np.argmax(data[:,0])
            gap = optimal_fid - data[pos_min_2,0]
            hx = (b2_array(pos_min,n_step)-0.5)*2.0
            print(T,'\t',n_step,'\t',optimal_fid,'\t',gap,'\t', gamma(hx),'\t',np.sum(hx))
            res[(round(T,2),n_step)]=[optimal_fid, gap, pos_min]
    with open('optimal.pkl','wb') as f:
        pickle.dump(res,f)















if __name__ == "__main__":
    main()
