import numpy as np
from tsne_visual import TSNE
from sklearn.decomposition import PCA
import sys
sys.path.append("..")
from utils import UTILS
from utils import parse_data
import pickle
from matplotlib import pyplot as plt



def main():
    utils=UTILS()
    parameters = utils.read_parameter_file(file="../para.dat")
    parameters['T']=3.0
    parameters['n_step']=250
    parameters['dt']=parameters['T']/parameters['n_step']

    file_name = utils.make_file_name(parameters,root="../data/")
    data = parse_data(file_name, v=3)

    fid_series = data['fid_series']
    print(len(fid_series))
    for s in fid_series:
        snp = np.array(s)
        plt.plot(range(len(s)),snp)
    
    plt.show()
    exit()




if __name__ == "__main__":
    main()