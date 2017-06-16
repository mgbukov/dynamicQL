


from utils import UTILS
import numpy as np
import pickle
from Hamiltonian import HAMILTONIAN
from quspin.operators import exp_op
import time,sys,os
from model import MODEL
from matplotlib import pyplot as plt
#from analysis.compute_observable import MB_observables
    
np.set_printoptions(precision=4)

def gamma(hx_tmp):
    n_step = len(hx_tmp)
    prod = hx_tmp[:n_step // 2]*(hx_tmp[n_step // 2:][::-1])
    pup = len(prod[prod > 0])
    return (1.*pup) / n_step


def main():
    # Utility object for reading, writing parameters, etc. 
    utils = UTILS()

    # Reading parameters from para.dat file
    parameters = utils.read_parameter_file()
    
    # Command line specified parameters overide parameter file values
    utils.read_command_line_arg(parameters,sys.argv)

    # Printing parameters for user
    utils.print_parameters(parameters)

    # Defining Hamiltonian
    H = HAMILTONIAN(**parameters)

    # Defines the model, and precomputes evolution matrices given set of states
    model = MODEL(H, parameters)

    #n_step = parameters['n_step']
    #X,y=sample_m0(10000,n_step,model)
    #print(y[0:10])
    #plt.hist(y,bins=20)
    #plt.show()
    
    rob_vs_T = {}
    n_eval = {}
    n_fid = {}

    for T_tmp in np.arange(0.025,10.001,0.025):
        for n_step in [100,200,400] :
            
            parameters['T']= T_tmp
            parameters['n_step'] = n_step
            parameters['dt'] = T_tmp/n_step

            file = utils.make_file_name(parameters,root='data/')
            
            with open(file,'rb') as f:
                _, data = pickle.load(f)
                n_elem = len(data)
                n_eval[(n_step,hash(T_tmp))]=[]
                n_fid[(n_step,hash(T_tmp))]=[]
                for elem in data:
                    n_eval[(n_step,hash(T_tmp))].append(elem[0])
                    n_fid[(n_step,hash(T_tmp))].append(elem[1])

    #print(n_eval)
    #exit()               
    n_eval_mean = {}
    n_fid_mean = {}
    for n_step in [100,200,400]:
        n_eval_mean[n_step]=[]
        n_fid_mean[n_step]=[]
        for T_tmp in np.arange(0.025,10.001,0.025):
            n_eval_mean[n_step].append([T_tmp,np.mean(n_eval[(n_step,hash(T_tmp))])/n_step])
            n_fid_mean[n_step].append([T_tmp,np.mean(n_fid[(n_step,hash(T_tmp))])])
    
    '''c_list=['#1a9850','#d73027','#c51b7d']
    for i, n_step in enumerate([100,200,400]):
        x = np.array(n_eval_mean[n_step])
        plt.plot(x[:,0], x[:,1],c='black',zorder=0)
        plt.scatter(x[:,0],x[:,1],c=c_list[i], marker='o', s=5, label='$N=%i$'%n_step,zorder=1)
    
    plt.title('Number of fidelity evaluations vs. ramp time \n for single flip')
    plt.ylabel('$N_{eval}/N$')
    plt.xlabel('$T$')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()'''

    c_list=['#1a9850','#d73027','#c51b7d']
    for i, n_step in enumerate([100,200,400]):
        x = np.array(n_fid_mean[n_step])
        plt.plot(x[:,0], x[:,1],c='black',zorder=0)
        plt.scatter(x[:,0],x[:,1],c=c_list[i], marker='o', s=5, label='$N=%i$'%n_step,zorder=1)
    
    plt.title('Mean fidelity vs. ramp time for single flip')
    plt.ylabel('$F$')
    plt.xlabel('$T$')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


    '''f_best = -1.
    s_best = -1
    n_symm_s = 2**(n_step//2)
    for s in range(n_symm_s):
        if s % 10000 == 0:
            print(s)
        model.update_protocol(b2_array_sym(s, n_step))
        f=model.compute_fidelity()
        if f > f_best :
            f_best = f
            s_best = s
            print(list(b2_array_sym(s_best, n_step)))
            print('f=%.14f'%f_best)

    print('best protocol: ')
    p=b2_array_sym(s_best, n_step)
    print(list(p))
    hx = ((p*1.0)-0.5)*2
    '''#print('m=',np.sum(hx))
    #print('g=',gamma(hx))
    #print('f=%.14f'%f_best)


def b2_array_sym(n10, n_step):
    return np.array(list(np.binary_repr(n10, width=n_step//2))+list(reversed(np.binary_repr(n10^(2**(n_step//2)-1), width=n_step//2))),dtype=np.int)

def sample_m0(n_sample, n_step, model:MODEL):
    X=np.empty((n_sample,n_step),dtype=int)
    y=np.zeros(n_sample)
    tmp = np.ones(n_step,dtype=int) # m = 0 state ...
    tmp[0:n_step//2] = 0
    np.random.shuffle(tmp)
    for i in range(n_sample):
        X[i]=tmp
        model.update_protocol(tmp)
        y[i]=model.compute_fidelity()
        np.random.shuffle(tmp)
    return X, y
        
def run_ES(parameters, model:MODEL, utils):
    
    n_step = parameters['n_step']
    n_protocol = 2**(n_step//2)
    exact_data = np.zeros((n_protocol,2), dtype=np.float64) # 15 digits precision
    
    b2_array = lambda n10 : np.array(list(np.binary_repr(n10, width=n_step//2)), dtype=np.int)
    st=time.time()
    # ---> measuring estimated time <---
    model.update_protocol(b2_array(0))
    psi = model.compute_evolved_state()
    model.compute_fidelity(psi_evolve=psi)
    model.compute_energy(psi_evolve=psi)
    print("Est. run time : \t %.3f s"%(0.5*n_protocol*(time.time()-st)))
    # ---> Starting real calculation <---

    st=time.time()
    for p in range(n_protocol):
        model.update_protocol(b2_array(p))
        psi = model.compute_evolved_state()
        exact_data[p] = (model.compute_fidelity(psi_evolve=psi), model.compute_energy(psi_evolve=psi))
    
    outfile = utils.make_file_name(parameters,root='data/')
    with open(outfile,'wb') as f:
        pickle.dump(exact_data, f, protocol=4)
    print("Total run time : \t %.3f s"%(time.time()-st))
    print("\n Thank you and goodbye !")
    f.close()

if __name__=='__main__':
    main()