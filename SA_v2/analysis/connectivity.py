from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from utils import UTILS
from utils import parse_data
import plotting
import scipy.spatial.distance as scidist

def mydist(x1,x2):
    return np.sum(np.abs(x1-x2))

def main():
    ut = UTILS()
    parameters=ut.read_parameter_file(file = "../para.dat")

    n_step = 100
    S_shannon =[]
    n_cluster= []
    slarge = []

    parameters['task']='SD'
    Trange = np.arange(0.05,1.0,0.01)
    n_fid = []
    n_visit = []
    fid = []
    std_fid = []
    energy = []
    #Trange = [3.6]
    n_best = 50000

    for T in Trange:
        #model = ut.quick_setup(argv=['T=%.3f'%T,'n_step=%i'%n_step],file='../para.dat')
        dt = T/n_step
        parameters['dt']=dt
        parameters['T'] = T
        parameters['n_step']= n_step
        file_name = ut.make_file_name(parameters,root="../data/")
        res = parse_data(file_name,v=2) # results stored here ...
        prot = res['protocol']
        
        asortF = np.argsort(res['F'])
        fid.append(np.mean(res['F'][asortF[-n_best:]]))
        std_fid.append(np.std(res['F']))
        n_visit.append(np.mean(res['n_visit'][asortF[-n_best:]]))
        n_fid.append(np.mean(res['n_fid'][asortF[-n_best:]]))
        energy.append(np.mean(res['E'][asortF[-n_best:]]))
        
        n_sample = prot.shape[0]
        #np.sum(-mag*np.log(mag))
        xunique, count = np.unique(prot, return_counts=True, axis=0)
        n_unique = len(xunique)
        #print(count)
        t=T
        print(T,'\t', n_unique,'\t',n_sample)
        '''v=scidist.squareform(scidist.pdist(xunique,mydist))
        print(scidist.squareform(scidist.pdist(xunique,mydist)))
        print([model.compute_fidelity(protocol = xi) for xi in xunique])
        for xi in xunique:
            plotting.protocol(range(150),xi)
        exit()


        exit()'''
        #[mydist(xunique[i],xunique[j]) for i in range(nunique) for j in range(nu
        #print(count)
        #print(len(count))
        #print(xunique)
        
        #plotting.protocol(range(150),xunique[0])
        #print(count)
        #_sample)
        #print(res['F'][0])
        #print('E0 :\t',model.compute_fidelity(protocol=xunique[0]))
        #print('E1 :\t',model.compute_fidelity(protocol=xunique[1]))
        #xtest = np.copy(xunique[1])
        #xtest[14]^=1
        #xtest[15]^=1
        #print(xtest)
        #print(xunique[0])
        #print("Einter :\t",model.compute_fidelity(protocol=xtest))
        #print(xunique)
        #exit()
        #plotting.protocol(range(50),xunique[1],title='F=%.10f'%model.compute_fidelity(protocol=xunique[1]))
        #print(count)
        #exit()
        #print(count)
        #exit()
        prob = (count*1.0) / np.sum(count)
        mag = np.mean(prot,axis=0)
        nzero = (mag > 1e-10)
        #mag[nzero]*np.log(mag[nzero])
        S_shannon.append(-np.sum(mag[nzero]*np.log(mag[nzero])))
        #plist2.append(prob)
        n_cluster.append(1.0*len(count)/n_sample)
        if len(count) == 1:
            slarge.append(1.0)
        else:
            asort=np.argsort(count)
            fraction = (1.0*count)/n_sample
            total_f= np.sum(fraction[fraction > 0.1])
            #largest_size = count[asort[-1]]
            slarge.append(np.max(count)/n_sample*1.0)
            #slarge.append(1.0*np.sum(count[asort[-5:]])/n_sample)
        #plist2.append(-np.sum(prob*np.log(prob))/np.log(n_sample))
        #print(T,'\t',-np.sum(prob*np.log(prob)))
    
    plt.scatter(Trange,S_shannon)
    plt.show()
    '''sorted_f = np.sort(res['F'])
    #print(np.std(sorted_f[-45000:]))
    #print(np.std(sorted_f))
    #exit()
    n_red = 41500
    plt.hist(np.sort(res['F'])[-n_red:], bins=100)
    n_sample = len(res['F'])
    plt.title("2-SF, T=%.2f, fidelity distribution ($N=%i$) \n considering the best %.1f %% of data"%(t,n_red,n_red/n_sample*100.))
    plt.show()
    exit()'''

    plt.scatter(Trange, n_fid)
    plt.title('nfid')
    plt.show()
    plt.scatter(Trange, n_visit)
    plt.title('nvisit')
    plt.show()
    plt.scatter(Trange, fid)
    plt.title('F')
    plt.show()
    plt.scatter(Trange, std_fid)
    plt.title('std F')
    plt.show()
    plt.scatter(Trange, energy)
    plt.title('E')
    plt.show()

    plt.scatter(Trange, slarge)
    plt.scatter(Trange, n_cluster)
    plt.title('Cluster size fraction and # of clusters')
    plt.show()

    plt.scatter(Trange, S_shannon)
    plt.title('Shannon entropy')
    plt.show()

    #plt.scatter(Trange,plist2)
    #plt.scatter(Trange,slarge)
    #plt.show()
    #plt.scatter(Trange,slarge)
    #plt.title('Shannon entropy of the local minima vs ramp time, 2-SF')
    #plt.xlabel('$T$')
    #plt.ylabel('$S/\log(N)$')
    #plt.show()



if __name__ == "__main__":
    main()