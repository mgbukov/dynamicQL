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


def paper_figure_1a():
    utils=UTILS()
    model = utils.quick_setup(argv = ['T=3.2'],file = "../para.dat")
    parameters = utils.read_parameter_file(file="../para.dat")
    parameters['T'] =3.2
    parameters['task']='SD'
    parameters['L']=6
    
    parameters['n_step']=200
    parameters['n_quench'] = 1000
    parameters['dt']=parameters['T']/parameters['n_step']


    file_name = utils.make_file_name(parameters,root="../data/")
    data = parse_data(file_name, v=3)
    
    fid_series = data['fid_series']
    fid = data['F']
    
    interval=[[0.,0.3],[0.3,0.9],[0.9,1.0]]

    c_location = []
    for sub_int in interval:
        c_location.append(np.where((fid < sub_int[1])&(fid > sub_int[0]))[0])

    c_idx = np.zeros(len(fid_series),dtype=int)
    for i,e in enumerate(c_location):
        c_idx[e]=i

    print('green:', np.count_nonzero(c_idx == 0))
    print('blue:', np.count_nonzero(c_idx == 1))
    print('red:', np.count_nonzero(c_idx == 2))
    #exit()

    green= (0.477789,0.719150,0.193583)
    blue = (0.229527,0.518693,0.726954)
    red = (0.797623,0.046473,0.127759)
    color_list = [green, blue, red]

    for i in range(3):
        plotting.density_trajectory_2(data['F'], color_list, c_location, show=False)
    
    plt.show()

    exit()
    plotting.trajectory(fid_series, c_idx)
    
    x=np.loadtxt('grapeprot.txt')
    plotting.protocol(np.linspace(0,3.2,100),x[:,0])
    plotting.protocol(np.linspace(0,3.2,100),x[:,1])
    plotting.protocol(np.linspace(0,3.2,100),x[:,2])
    #n_step = 200
    
    print(x)
    
    # --------------------------------
    # --------------------------------

    exit()
    fid = data['F']
    print(np.mean(fid))
    protocols =  data['protocol']
    interval = [0.,1.0]
    protocols = protocols[(fid<interval[1]) & (fid>interval[0])] # protocols within an interval
    fid_res = fid[(fid<interval[1]) & (fid>interval[0])]

    for s in fid_series:
        snp = np.array(s)
        plt.plot(range(len(s)),snp,color='blue')
        plt.title('T=%.2f, nStep=%i'%(parameters['T'],parameters['n_step']))
        plt.tight_layout()
    
    plt.show()
    exit()
    plt.savefig('plots/SD_fidseries_T=%.2f_nStep=%i.pdf'%(parameters['T'],parameters['n_step']))
    plt.clf() 



def main():
    paper_figure_1a()


    exit()
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