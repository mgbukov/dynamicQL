'''
Created on Mar 7 , 2017

@author: Alexandre Day

Purpose:

    This module is for performing stochastic descent
    in magnetization sectors. Seed to the SD optimizer
    can be chosen to be found via simulated annealing withing 
    the m=0 sector

'''

import LZ_sim_anneal as LZ
import time
import numpy as np
import sys
from matplotlib import pyplot as plt
import plotting
import pickle

def main():

    if len(sys.argv) > 1:
        _, L, dt, n_step, m, n_sample = sys.argv
        L = int(L); n_step = int(n_step); m = int(m)
        dt = float(dt) ; n_sample = int(n_sample)
    else:
        L = 6
        dt = 0.01
        n_step = 250
        m = 0
        n_sample = 1

    param = {'L' : L, 'dt': dt, 'n_step': n_step, 'm': m}
    file_name = make_file_name(param, root= "data/")
    print("running : ",file_name)

    if L == 1:
        div = 2.
    else:
        div = 1.
    custom_prot=LZ.custom_protocol(
        J=1.0, hz=1.0, hx_init_state=-2.0/div, hx_target_state=2.0/div,
        L=L, delta_t=dt, 
        hx_i=-4., hx_max=4., action_set_=[-8.,0.,8.], option='fast')

    Ti = 1e-3
    N_quench = 20000
    sample_result = []
    print("Running for %i samples"%n_sample)

    for _ in range(n_sample):
        fid_best, hx_tmp, fid_best_list, hx_tmp_list = SA(Ti, N_quench, n_step, m, custom_prot, init_random = True, info = True)
        #sample_result.append([fid_best, hx_tmp, fid_best_list, hx_tmp_list])
        sample_result.append([fid_best, hx_tmp])
    
    with open(file_name,'wb') as f:
        pickle.dump(sample_result,f)

    #fid_best_list=np.array(fid_best_list)
    #plt.scatter(fid_best_list[:,0],fid_best_list[:,1])
    #plt.show()

    #Gamma=gamma(hx_tmp)
    #plotting.protocol(np.arange(0.,n_step)*dt, hx_tmp, 
    #    title = "$m=%i$, $\Gamma=%.4f$, $F=%.12f$,\n $L=%i$, $dt=%.4f$, $T=%.3f$"%(m,Gamma,fid_best,L,dt,dt*n_step))

def gamma(hx_tmp):
    n_step = len(hx_tmp)
    prod = hx_tmp[:n_step // 2]*(hx_tmp[n_step // 2:][::-1])
    pup = len(prod[prod > 0])
    return (1.*pup) / n_step

def swap(hx,x1,x2):
    tmp = hx[x2]
    hx[x2] = hx[x1]
    hx[x1] = tmp

def SD(N_step, sector, custom_prot_obj, init_random = True, init_state=None, max_depth = 1000, info = True, n_tot_eval = 0):
    """
    Stochastic descent in a particular magnetization sector
    """

    system = custom_prot_obj
    # perform random permutation
    n_up = int( (N_step + sector) / 2)

    hx_tmp = np.array([4]*n_up + [-4]*(N_step-n_up)) # initialize sector

    if init_random is True:
        np.random.shuffle(hx_tmp)
    if init_state is not None:
        hx_tmp = init_state

    fid_best = system.evaluate_protocol_fidelity(hx_tmp)
    
    n_tot_eval = n_tot_eval
    fid_best_list=[]
    hx_tmp_list=[]
    
    print("--> Starting Stochastic Descent !")
    print("{0:<6}{1:<9}{2:<20}{3:<10}".format("#","n_eval","Fidelity","Gamma"))
    is_minima = True
    x1_ar, x2_ar = np.triu_indices(N_step,1)
    n_permutation = x1_ar.shape[0]
    order = np.arange(0, n_permutation ,dtype=np.int)

    for _ in range(max_depth):

        np.random.shuffle(order)
        for pos, count in zip(order,range(n_permutation)):

            x1, x2 = (x1_ar[pos], x2_ar[pos])
            swap(hx_tmp,x1,x2)
            fid_new = system.evaluate_protocol_fidelity(hx_tmp)
            n_tot_eval +=1

            if (fid_new > fid_best) :
                
                fid_best_list.append([n_tot_eval,fid_new])
                hx_tmp_best = np.copy(hx_tmp)
                hx_tmp_list.append(hx_tmp_best)

                fid_best = fid_new

                if info:
                    swap(hx_tmp,x1,x2)
                    g1 = 0 if hx_tmp[x1]*hx_tmp[-(x1+1)] < 0 else 1
                    g2 = 0 if hx_tmp[x2]*hx_tmp[-(x2+1)] < 0 else 1
                    swap(hx_tmp,x1,x2)
                    out_str = "{0:<6}{1:<9}{2:<20.14f}{3:<10.5f}{4:10}{5:10}".format(_,n_tot_eval,fid_best, gamma(hx_tmp_best),g1,g2)
                    print(out_str)
                break
            else: # reject move
                swap(hx_tmp,x1,x2)
            
        if count == n_permutation - 1 :
            print("--> Minima reached !")
            out_str = "{0:<6}{1:<9}{2:<20.14f}{3:<10.5f}{4:10}{5:10}".format(_,n_tot_eval,fid_best, gamma(hx_tmp_best),g1,g2)
            print(out_str)
            break

    return fid_best, hx_tmp_best, fid_best_list, hx_tmp_list


def SA(Ti, N_quench, N_step, sector, custom_prot_obj, init_random = True, info = True):
    """
    Simulated annealing in a particular magnetization sector
    """

    system = custom_prot_obj
    # perform random permutation
    n_up = int( (N_step + sector) / 2)
    fid_best = -10.

    hx_tmp = np.array([4]*n_up + [-4]*(N_step-n_up)) # initialize sector

    if init_random is True:
        np.random.shuffle(hx_tmp)

    fid_old = system.evaluate_protocol_fidelity(hx_tmp)
    n_refuse = 0
    n_tot_eval = 0
    fid_best_list=[]
    hx_tmp_list=[]
    T=Ti

    print("--> Starting Simulated Annealing !")
    print("{0:<6}{1:<9}{2:<20}{3:<10}".format("#","n_eval","Fidelity","Gamma"))

    while T > 0.:

        beta = 1./T
        x1 = np.random.randint(N_step)
        x2 = np.random.randint(N_step)
        swap(hx_tmp,x1,x2)
        fid_new = system.evaluate_protocol_fidelity(hx_tmp)
        n_tot_eval += 1
        
        if fid_new > fid_best :
            fid_best_list.append([n_tot_eval,fid_new])
            fid_best = fid_new 
            hx_tmp_best = np.copy(hx_tmp)

        dF = fid_new - fid_old
        if dF > 0 : # accept move
            n_refuse = 0
            fid_old = fid_new
        elif np.random.random() < np.exp(beta*dF): # accept move with temperature
            n_refuse = 0
            fid_old = fid_new
        else: # revert move (reject !)
            swap(hx_tmp,x1,x2)
            n_refuse +=1

        T = Ti*(1.-(1.*n_tot_eval)/N_quench)

        if n_tot_eval % 1000 == 0:
            hx_tmp_list.append(hx_tmp_best)
            out = "{0:<6}{1:<9}{2:<20.14f}{3:<10.5f}".format('___', n_tot_eval,fid_best,gamma(hx_tmp_best))
            print(out)

    print('\n\n')

    fid_best, hx_tmp_best, fid_best_list, hx_tmp_list= SD(
        N_step, sector, custom_prot_obj,
        init_state=hx_tmp_best, n_tot_eval = n_tot_eval,
        info = info
        )
    
    return fid_best, hx_tmp_best, fid_best_list, hx_tmp_list

def make_file_name(param, root="",ext=".pkl"):
    key_format = {
        'L':'{:0>2}',
        'dt':'{:.4f}',
        'n_step':'{:0>4}',
        'm':'{:0>3}'
    }

    f = [k+"-"+key_format[k].format(param[k]) for k in sorted(key_format)]
    return root+'SA_'+"_".join(f)+ext



if __name__ == "__main__":
    main()