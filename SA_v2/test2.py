import pickle
import numpy as np
from utils import UTILS
import pandas as pd
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def main():

    with open('optimal.pkl','rb') as f:
        res = pickle.load(f)
    ### READING ES data and finding optimal state and the gap to excited states.
    ### Looping over parameters and storing data in a dictionary. 
    
    utils=UTILS()
    parameters = utils.read_parameter_file()
    
    '''prob_vs_T = {}
    for T in [0.1, 0.3, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] :
        prob_vs_T[T] = np.zeros((13,2),dtype=np.float32)
        ii = 0
        for n_step in [4,6,8,10,12,14,16,18,20,22,24,26,28] :
            
            parameters['T']= T
            parameters['n_step'] = n_step
            parameters['dt'] = T/n_step
            gs_fid = res[(T,n_step)][0]
            
            file = utils.make_file_name(parameters,root='data/')
            
            with open(file,'rb') as f:
                _, data = pickle.load(f)
                n_elem = len(data)
                v = np.zeros(n_elem,dtype=np.float32) # careful here, precision is important ---> !!!
                for i,elem in zip(range(n_elem),data):
                    v[i] = elem[1]
            prob = (v[np.abs(v - gs_fid) < 1e-14].shape[0])/n_elem
            
            prob_vs_T[T][ii,0] = n_step
            prob_vs_T[T][ii,1] = prob 
            print(T,' ',n_step,' : ',prob)
            ii += 1'''

    with open('scaling.pkl', 'rb') as f:
        #pickle.dump(prob_vs_T,f)
    #exit()
        prob_vs_T = pickle.load(f)

    for T in [0.1, 0.3, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] :
        plt.plot(prob_vs_T[T][:,0],-np.log(prob_vs_T[T][:,1]),label='$T=%.2f$'%T)
        plt.scatter(prob_vs_T[T][:,0],-np.log(prob_vs_T[T][:,1]))

    plt.xlabel('$N$',fontsize=16)
    plt.ylabel('$-\log p(h(t)=h_{\mathrm{opt}}(t))$',fontsize=16)
    plt.legend(loc='best')
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()