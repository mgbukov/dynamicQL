from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from utils import UTILS
from utils import parse_data
import plotting
from compute_observable import Ed_Ad_OP
from compute_observable import Ed_Ad_OP_2


def main():

    ut = UTILS()
    parameters=ut.read_parameter_file(file = "../para.dat")
    parameters['L']=1

    n_step = 400
    nrange = np.arange(10,800,10,dtype=int)
    Trange = np.arange(0.05,4.01,0.05)
    dt = 0.005
    parameters['dt'] = dt
    mean_fid = []
    std_fid = []
    n_fid = []
    ed1 = []
    ed2 = []

    for n in nrange:
    #for n in nrange:
        #model = ut.quick_setup(argv=['T=%.3f'%T,'n_step=%i'%n_step],file='../para.dat')
        parameters['T'] = n*dt
        #parameters['T'] = T
        parameters['n_step']= n
        #parameters['n_step']= n_step
        #parameters['dt'] = parameters['T']/parameters['n_step']
        file_name = ut.make_file_name(parameters,root="../data/")
        res = parse_data(file_name) # results stored here ...
        print(n,'\t',len(res['F']))
        mean_fid.append(np.max(res['F']))
        n_fid.append(np.mean(res['n_fid']))
        std_fid.append(np.std(res['F']))
        tmp = 8*(res['protocol'] - 0.5)
        ed1.append(Ed_Ad_OP(tmp))
        #ed2.append(Ed_Ad_OP_2(res['protocol'],min_h=0, max_h=1))
    
    plt.plot(nrange*dt,n_fid ,label='ed1')
    #plt.plot(nrange*dt, ed2,label='ed2')
    plt.legend(loc='best')
    plt.show()
    exit()
    n_step = len(res['protocol'][0])
    plotting.protocol(Trange,np.mean(res['protocol'],axis=0),title='$T=%.3f$'%(dt*n_step))
    #plotting.protocol(np.arange(0,n_step)*dt,np.mean(res['protocol'],axis=0),title='$T=%.3f$'%(dt*n_step))
    #plt.scatter(nrange*dt,std_fid)
    #plt.plot(nrange*dt,std_fid)
    plt.show()
if __name__ == "__main__":
    main()