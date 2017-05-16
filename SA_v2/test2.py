import copy
import numpy as np
import time 
import itertools
from utils import UTILS
import pandas as pd
import pickle

def main():
    utils = UTILS()
    parameters = utils.read_parameter_file()

    for n_step in range(10,401,10):
        parameters['n_step'] = n_step
        res={}
        for n_quench in [100,500,1000,2000,5000,10000]:
            parameters['n_quench']=n_quench
            file = utils.make_file_name(parameters,root="data/")
            with open(file,'rb') as f:
                _, data = pickle.load(f)
                res[(n_step,n_quench)] = np.mean(pd.DataFrame(data).iloc[:,1])
    
    
    #print(res[100].iloc[189,3])
    for n_quench in [100,500,1000,2000,5000,10000]:
        print(np.mean(res[n_quench].iloc[:,1]))




    #[['n_step',range(10,401,10)],['n_quench',[100,500,1000,2000,5000,10000]]
    #['n_step',range(10,41,10)],['Ti',np.arange(0.2,0.3,0.01)]
    #print(utils.make_file_name(parameters))






if __name__ == "__main__":
    main()