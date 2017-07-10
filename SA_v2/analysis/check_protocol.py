from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from utils import UTILS
from utils import parse_data
import plotting

def main():

    ut = UTILS()
    parameters=ut.read_parameter_file(file = "../para.dat")
    parameters['L']=2

    nrange = np.arange(4,200,4,dtype=int)
    nrange = [nrange[18]]
    dt = 0.02
    parameters['dt'] = dt
    mean_fid = []
    std_fid = []

    for n in nrange:
        #model = ut.quick_setup(argv=['T=%.3f'%T,'n_step=%i'%n_step],file='../para.dat')
        parameters['T'] = n*dt
        parameters['n_step']= n
        file_name = ut.make_file_name(parameters,root="../data/")
        res = parse_data(file_name,v=2) # results stored here ...
        mean_fid.append(np.mean(res['F']))
        std_fid.append(np.std(res['F']))
        prot = res['protocol']
    
    n_step = len(res['protocol'][0])
    plotting.protocol(np.arange(0,n_step)*dt,np.mean(res['protocol'],axis=0),title='$T=%.3f$'%(dt*n_step))
    #plt.scatter(nrange*dt,std_fid)
    #plt.plot(nrange*dt,std_fid)
    plt.show()
if __name__ == "__main__":
    main()