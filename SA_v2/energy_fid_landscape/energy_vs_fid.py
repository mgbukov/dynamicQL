import sys
sys.path.append("..")
from utils import UTILS
import pickle
from matplotlib import pyplot as plt
#import seaborn as sns
from fdc.plotting import density_map
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import analysis.plotting as plt2
from model import MODEL
from Hamiltonian import HAMILTONIAN
from fdc import KDE

def bool2int(x):
    y = 0
    for i,j in enumerate(x[::-1]):
        y += j<<i
    return y

def excitations(Fspectrum, parameters):
    
    hopt_n10=np.argmax(Fspectrum) 

    print('L','T','dt',sep='\t')
    print(parameters['L'],parameters['T'],parameters['dt'],sep='\t')
    print('Optimal fidelity : %.8f'%Fspectrum[hopt_n10])
    
    # Defining Hamiltonian
    H = HAMILTONIAN(**parameters)

    # Defines the model, and precomputes evolution matrices given set of states
    model = MODEL(H, parameters)

    n_step=parameters['n_step']

    b2_array = lambda n10 : np.array(list(np.binary_repr(n10, width=n_step)), dtype=np.int)

    exc_dict = {} # sorting out excitations by they m and m_z

    hopt = b2_array(hopt_n10) # optimal protcol in array base 2
    mag_opt = np.sum(hopt) # same as spin 1/2 ... -> excitations are integer excitations, called spin-1 and spin-2 excitations

    for i in range(n_step): # spin 1 excitations
        hopt[i]^=1
        mz = np.sum(hopt) - mag_opt
        if (1, mz) not in exc_dict.keys():
            exc_dict[(1, mz)]=[]
        
        exc_dict[(1, mz)].append(bool2int(hopt))
        hopt[i]^=1
    
    # --------------------------------
    for i in range(n_step): # spin 2 excitations
        for j in range(i):

            hopt[i]^=1
            hopt[j]^=1

            mz = np.sum(hopt) - mag_opt
            if (2, mz) not in exc_dict.keys():
                exc_dict[(2, mz)]=[]
            exc_dict[(2, mz)].append(bool2int(hopt))
            
            hopt[i]^=1
            hopt[j]^=1
    
    return exc_dict

def main():
    import os
    
    utils=UTILS()
    
    parameters=utils.read_parameter_file(file="../para.dat")

    file_bw = 'bandwidth_L=6_nStep=20.pkl'
    if os.path.isfile(file_bw):
        with open(file_bw,'rb') as f:
            bandwidths = pickle.load(f)
    else:
        bandwidths = {}
    
    for T in np.arange(0.3, 4.01, 0.1):
        print("density map for\t", T)
        parameters['task']='ES'
        parameters['L'] = 6
        parameters['n_step']= 20
        parameters['T'] = T
        parameters['dt'] = parameters['T']/parameters['n_step']
        parameters['n_quench'] = 1000

        file = utils.make_file_name(parameters, root="../data/")

        with open(file,'rb') as f:
            data = pickle.load(f)
            n_sample = data.shape[0]
    
        exc_dict = excitations(data[:,0], parameters)

        X = np.column_stack((data[:, 0], data[:, 1])) # fidelity vs energy

        Emin = np.min(X[:,1])
        arg_Fmax = np.argmax(X[:,0])
        Fmax = X[arg_Fmax, 0]

        print("--> density estimate on : ", X.shape)

        # transformations, so we can visualize the high-fidelity regions 
        X[:,0]=-np.log(X[:,0])/parameters['L']
        X[:,1]=(X[:,1]-Emin)/parameters['L']

        if round(T,3) in bandwidths.keys():
            kde = KDE(extreme_dist=True,bandwidth=bandwidths[round(T,3)])
        else:
            kde = KDE(extreme_dist=True)

        kde.fit(X) # determining bandwidth + building model
        bandwidths[round(T,3)] = kde.bandwidth

        #with open('kde_tmp.pkl','wb') as f:
        #    pickle.dump(kde, f)
    
    # Just make some prelim plots, fidelity, etc. Let's see what we get for now. 
    # -------------->
        print('plotting')  
        plt2.density_map(X, kde, xlabel='$-\log F/L$',ylabel='$(E-E_0)/L$', show=False, n_mesh=400)
        print('excitations')
        plot_excitations(X, exc_dict, a_fmax=arg_Fmax)

        #plt.title('$T=%.2f, N=%i $'%(parameters['T'], parameters['n_step']))
        plt.tight_layout(pad=0.2)
        plt.legend(loc='best')
        plt.savefig('ES_L=%i_T=%.2f_nStep=%i.pdf'%(parameters['L'], parameters['T'], parameters['n_step']))

        with open('bandwidth_L=%i_nStep=%i.pkl'%(parameters['L'],parameters['n_step']),'wb') as f:
            pickle.dump(bandwidths, f)
            f.close()
        exit()
    
def plot_excitations(X, exc_dict, a_fmax = None):

    # ----------
    marker_dict = {
        (1,1): 'o',
        (1,-1): 'o',
        (2,-2): 'd',
        (2,2) : 'd',
        (2,0) : 'd'
    }
    # ----------
    color_dict = {
        (1,1): 'lawngreen',
        (1,-1): 'red',
        (2,-2): 'magenta',
        (2,2) : 'cyan',
        (2,0) : 'yellow'
    }
    # ----------
    label_dict = {
        (1,1): '$m=1$',
        (1,-1): '$m=-1$',
        (2,-2): '$m=2$',
        (2,2): '$m=-2$',
        (2,0): '$m=0$'
    }
    # ----------
    
    ''' tmp_dict = {
        (1,1) : exc_dict[(1,1)],
        (1,-1) : exc_dict[(1,-1)],
        (2,-2) : exc_dict[(2,-2)],
        (2,2) : exc_dict[(2,2)],
        (2,0) : exc_dict[(2,0)]
    } '''

    list_key = [(1,1),(1, -1),(2, -2),(2, 2), (2,0)]

    for k in list_key:
        v = exc_dict[k]
        #print(v)
        Xtmp = X[v]
        plt.scatter(Xtmp[:,0], Xtmp[:,1], c=color_dict[k],marker=marker_dict[k],s=30,label=label_dict[k],zorder=3,edgecolor='black',linewidths=0.5)
    
    if a_fmax is not None:
        plt.scatter([X[a_fmax,0]],[X[a_fmax,1]],c='orange',marker='*', s=60, zorder=3,label='$F_{\mathrm{optimal}}$', edgecolor='black',linewidths=0.5)







def transform_exc(data, exc, emin):
    X=data[exc]
    X[:,0] = np.log(X[:,0])
    X[:,1] = X[:,1] - emin
    return X

if __name__ == "__main__":
    main()
