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
    parameters['T']=3.4
    parameters['n_step']=200
    parameters['n_quench'] = 6666
    parameters['dt']=parameters['T']/parameters['n_step']

    file_name = utils.make_file_name(parameters,root="../data/")
    data = parse_data(file_name, v=3)
    
    fid_series = data['fid_series']
    protocols = data['protocol']
    fid = data['F']
    protocols = protocols[(fid<interval[1]) & (fid>interval[0])] # protocols within an interval
    fid_res = fid[(fid<interval[1]) & (fid>interval[0])]
    #plt.hist(fid_res,bins=100)
    #plt.show()
    #exit()
    meanprot = np.mean(protocols, axis = 0)
    tprot = np.linspace(0,parameters['T'],parameters['n_step'])
    plotting.protocol(tprot,np.round(meanprot), show=True)
    exit()
    print("Continuous protocol fidelity:\t",model.compute_continuous_fidelity(meanprot))
    print("Discrete protocol fidelity:\t",model.compute_fidelity(protocol=np.round(meanprot)))
    print(compute_observable.Ed_Ad_OP(protocols, min_h =0, max_h = 1))

    exit()
    plotting.protocol(tprot,meanprot,show=True)

    #print(protocols)
    exit()

def make_histogram(parameters, version = 2):
    utils=UTILS()
    file_name = utils.make_file_name(parameters,root="../data/")
    data = parse_data(file_name, v=version)
    fidelities = data['F']
    plt.hist(fidelities, bins = 200)
    plt.savefig('histogram/hist.pdf')
    #plt.show()

    #plotting.protocol(np.mean(protocols)
    #print(len(fid_series))
'''     for s in fid_series:
        snp = np.array(s)
        plt.plot(range(len(s)),snp)
        plt.title('T=%.2f, nStep=%i'%(parameters['T'],parameters['n_step']))
        plt.tight_layout()
    
        plt.savefig('plots/SD_fidseries_T=%.2f_nStep=%i.pdf'%(parameters['T'],parameters['n_step']))
        plt.clf() '''

def plot_series(parameters):
    Trange =np.arange(0.1,4.01,0.1)
    for t in Trange:
        parameters['T']=t
        parameters['n_step']=200
        parameters['n_quench'] = 6666
        parameters['dt']=parameters['T']/parameters['n_step']

        file_name = utils.make_file_name(parameters,root="../data/")
        data = parse_data(file_name, v=3)

        fid_series = data['fid_series']
        print(len(fid_series))
        for s in fid_series:
            snp = np.array(s)
            plt.plot(range(len(s)),snp)
            plt.title('T=%.2f, nStep=%i'%(parameters['T'],parameters['n_step']))
            plt.tight_layout()
        
        plt.savefig('plots/SD_fidseries_T=%.2f_nStep=%i.pdf'%(parameters['T'],parameters['n_step']))
        plt.clf()


if __name__ == "__main__":
    main()