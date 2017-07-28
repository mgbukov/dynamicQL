import numpy as np
from tsne_visual import TSNE
from sklearn.decomposition import PCA
import sys
sys.path.append("..")
from utils import UTILS
from utils import parse_data
import pickle
from matplotlib import pyplot as plt
import plotting
import compute_observable


def main():
    utils=UTILS()
    model = utils.quick_setup(argv = ['T=3.3'],file = "../para.dat")
    parameters = utils.read_parameter_file(file="../para.dat")
    Trange =np.arange(0.1,4.01,0.1) # maybe trick is to bin stuff up --> ?!
    interval = [0.85,1.0]
    #for t in Trange:
    parameters['T']=3.2
    parameters['n_step']=100
    parameters['n_quench'] = 1000
    parameters['dt']=parameters['T']/parameters['n_step']

    file_name = utils.make_file_name(parameters, root="../data/")
    data = parse_data(file_name, v=3)
    
    fid_series = data['fid_series']
    protocols = data['protocol']
    fid = data['F']
    protocols = protocols[(fid<interval[1]) & (fid>interval[0])] # protocols within an interval
    

def make_histogram(parameters, version = 2):
    utils=UTILS()
    file_name = utils.make_file_name(parameters,root="../data/")
    data = parse_data(file_name, v=version)
    fidelities = data['F']
    plt.hist(fidelities, bins = 200)
    plt.savefig('histogram/hist.pdf')

