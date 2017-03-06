'''
Created on Jan 2 , 2017

@author: Alexandre Day

Purpose:
    This module is just for testing protocols

'''

import LZ_sim_anneal as LZ
import time
import numpy as np
import sys
from matplotlib import pyplot as plt
import plotting

def main():
    L=6
    custom_prot=LZ.custom_protocol(
        J=-1.0,
        L=L, hz=1.0, hx_init_state=-2.0, hx_target_state=2.0,
        delta_t=0.0025, hx_i=-4., hx_max=4., action_set_=[-8.,0.,8.],
        option='fast'
    )

    hx_list=[]
    fid_list=[]
    symm_list=[]
    
    n_step_list=[1300,100,150,200,300,400,450,500,550,600,650,700,800]
    for nstep in n_step_list:
        m = 8
        print(nstep)
        fid_best, hx_tmp, fid_best_list = SD(nstep, m, custom_prot, n_refuse_max=10000, n_random_restart=1, n_eval_max = 200000, init_random = True)
        print(fid_best,'\t',check_symmetry(hx_tmp))
        #print(check_symmetry(hx_tmp))
        hx_list.append(hx_tmp)
        fid_list.append(fid_best)
        symm_list.append(check_symmetry(hx_tmp))
        plotting.protocol(np.array(range(len(hx_tmp)))*0.0025,hx_tmp,
        title="$m=%i$, $L=%i$, $nstep=%i$, $T=%.3f$, $F=%.6f$"%(m,L,nstep,nstep*0.0025,fid_best),
        show=False, out_file="m=0_nstep=%i.pdf"%nstep)
        fid_best_list=np.array(fid_best_list)
        exit()

    plt.scatter(np.array(n_step_list)*0.0025, fid_list)
    plt.scatter(np.array(n_step_list)*0.0025, symm_list)
    plt.show()
    exit()
    m=0
    fid_res = []

    symm_res = []
    for t in range(50,1050,50):
        print(t)
        fid_best, hx_tmp = SD(t, m, custom_prot, n_refuse_max=5000, n_random_restart=1)
        fid_res.append(fid_best)
        symm_res.append(check_symmetry(hx_tmp))
    
    plt.scatter(range(50,1050,50),fid_res)
    plt.scatter(range(50,1050,50),symm_res)

    m=2
    fid_res = []
    symm_res = []
    for t in range(50,1050,50):
        print(t)
        fid_best, hx_tmp = SD(t, m, custom_prot, n_refuse_max=5000, n_random_restart=1)
        fid_res.append(fid_best)
        symm_res.append(check_symmetry(hx_tmp))

    plt.scatter(range(50,1050,50),fid_res)
    plt.scatter(range(50,1050,50),symm_res)

    plt.show()


    exit()
    # --------->  
    mag_vs_fid=[]

    m = 0
    fid_best, hx_tmp = SD(650, m, custom_prot, n_refuse_max=5000, n_random_restart=1)
    print(custom_prot.evaluate_protocol_fidelity(hx_tmp))
    print("symm:",check_symmetry(hx_tmp))

    m = 2
    fid_best, hx_tmp = SD(650, m, custom_prot, n_refuse_max=5000, n_random_restart=1)
    print(custom_prot.evaluate_protocol_fidelity(hx_tmp))
    print("fid, symm, m= ",fid_best,'\t',check_symmetry(hx_tmp),'\t',m)

    m = 0
    fid_best, hx_tmp = SD(800, m, custom_prot, n_refuse_max=5000, n_random_restart=1)
    print(custom_prot.evaluate_protocol_fidelity(hx_tmp))
    print("symm:",check_symmetry(hx_tmp))

    m = 2
    fid_best, hx_tmp = SD(800, m, custom_prot, n_refuse_max=5000, n_random_restart=1)
    print(custom_prot.evaluate_protocol_fidelity(hx_tmp))
    print("fid, symm, m= ",fid_best,'\t',check_symmetry(hx_tmp),'\t',m)
    
    exit()

    #plotting.protocol(range(len(hx_tmp)),hx_tmp)
    #exit()
    #print(np.sum(hx_tmp)/4)
    #exit()


    for m in [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28]:

        fid_best, hx_tmp = SD(650, m, custom_prot, n_refuse_max=10000, n_random_restart=1) # no need for restart ?
        mag_vs_fid.append([m,fid_best])
        print("Mag,best = %i, %.6f"%(m,fid_best))

    mag_vs_fid=np.array(mag_vs_fid)
    plt.scatter(mag_vs_fid[:,0], mag_vs_fid[:,1])
    plt.show()

    #print(np.sum(hx_tmp))

    #plotting.protocol(range(150), hx_tmp)

    print(" -------- ")
    exit()

'''mag_list=[0,2,4,6,8,10,12,14,16,18,20,22]
best_fid=[]
for magnetization in mag_list:
    print(magnetization)
    fid=[]
    N_step=200
    n_up = int((magnetization + N_step) / 2)
    hx_tmp=np.array([4]*n_up + [-4]*(N_step-n_up)) # anneal this ! 
    for i in range(4000):
        #hx_tmp=[[-4,4][np.random.randint(2)] for i in range(N_step)]
        np.random.shuffle(hx_tmp)
        fid.append(custom_prot.evaluate_protocol_fidelity(hx_tmp))
        #np.random.shuffle(hx_tmp)
    best_fid.append(np.mean(fid))'''

def check_symmetry(hx_tmp):
    n_step = len(hx_tmp)
    prod = hx_tmp[:n_step // 2]*(hx_tmp[n_step // 2:][::-1])
    pup = len(prod[prod > 0])
    return (1.*pup) / n_step

def swap(hx,x1,x2):
    tmp = hx[x2]
    hx[x2] = hx[x1]
    hx[x1] = tmp

def SD(N_step, sector, custom_prot_obj, n_refuse_max=10, n_random_restart=50, n_eval_max = 10000, init_random = True, LR_symm=False):
    system = custom_prot_obj
    # perform random permutation
    n_up = int( (N_step + sector) / 2)
    
    fid_best = -10.

    for i in range(n_random_restart): # random restarts
        hx_tmp=np.array([4]*n_up + [-4]*(N_step-n_up)) # initialize sector
        if init_random is True:
            np.random.shuffle(hx_tmp)
        
        fid_old = system.evaluate_protocol_fidelity(hx_tmp)
        n_refuse = 0
        n_tot_eval = 0
        fid_best_list=[]

        while True:

            x1 = np.random.randint(N_step)
            x2 = np.random.randint(N_step)
            swap(hx_tmp,x1,x2)
            fid_new = system.evaluate_protocol_fidelity(hx_tmp)
            n_tot_eval += 1
            if n_tot_eval > n_eval_max:
                break
            if n_tot_eval % 1000 == 0:
                print(n_tot_eval,'\t',fid_best)

            if fid_new > fid_best :
                fid_best_list.append([n_tot_eval,fid_new])
                fid_best = fid_new 
                hx_tmp_best = np.copy(hx_tmp)

            if fid_new > fid_old : # accept move
                n_refuse = 0
                fid_old = fid_new
            else: # reject move
                swap(hx_tmp,x1,x2)
                n_refuse +=1
            
            if n_refuse > n_refuse_max: # number of random permutations refused before considering this is a local minima !
                break
    return fid_best, hx_tmp, fid_best_list

if __name__ == "__main__" :
    main()