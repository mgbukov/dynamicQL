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
import pickle
import os.path
import seaborn as sns
from SD_msector import make_file_name

def b2(n10,w=10):
    x = np.array(list(np.binary_repr(n10, width=w)),dtype=np.int)
    x[x < 1] = -4
    x[x > 0] = 4
    return x

def main():

    L=6
    dt = 0.005
    m = 0
    n_step = 100

    param = {'L' : L, 'dt': dt, 'n_step': n_step, 'm': m}
    file_name = make_file_name(param, root= "data/")

    custom_prot=LZ.custom_protocol(
        J=1.0,
        L=L, hz=1.0, hx_init_state=-2.0, hx_target_state=2.0,
        delta_t=dt, hx_i=-4., hx_max=4., action_set_=[-8.,0.,8.],
        option='fast'
    )

    choice = 2
    count = -1
    palette = np.array(sns.color_palette('hls',10))
    n_step = 330
    dt=0.01
    param = {'L' : L, 'dt': dt, 'n_step': n_step, 'm': m}
    file_name = make_file_name(param, root= "data/")
    with open(file_name,'rb') as f:
        data=pickle.load(f)
    #print(data[4][0])
    #print(data[4][1])
    #print(gamma(data[4][1]))
    plotting.protocol(range(330),data[4][1])
    exit()
    for d in data:
        print(d[0],'\t',gamma(d[1]))
    exit()

    for m in [0]:
        count+=1
        best_fid_all = []
        mean_time_all = []
        gamma_all = []        
        n_step_all = []

        for n_step in np.arange(10,401,10):
            L = 6
            dt = 0.01
            #m = 2
            #n_step = 100

            param = {'L' : L, 'dt': dt, 'n_step': n_step, 'm': m}
            file_name = make_file_name(param, root= "data/")

            if os.path.exists(file_name):
                n_step_all.append(n_step)
                with open(file_name,'rb') as f:
                    result = pickle.load(f)
                
                eval_times = []
                symm_all = []
                fid_all = []

                for sample in result :
                    [fid_best, hx_tmp, fid_best_list, hx_tmp_list] = sample
                    eval_times.append(fid_best_list[-1][0])
                    symm_all.append(gamma(hx_tmp))
                    fid_all.append(fid_best)
            
                print(" n = %i "%n_step)
                print(max(fid_all),'\t\t', np.mean(fid_all))
                print(np.mean(eval_times),'\t\t', np.std(eval_times))
                print(min(symm_all),'\t\t', np.mean(symm_all))
                print("\n")
                best_fid_all.append(np.max(fid_all))
                mean_time_all.append(np.mean(eval_times))
                gamma_all.append(np.min(symm_all))

        if choice == 0:
            plt.plot(np.array(n_step_all)*dt,best_fid_all, c=palette[count],label='$m=%i$'%m)
        elif choice == 1:
            plt.plot(np.array(n_step_all)*dt,mean_time_all, marker='o',label='$m=%i$'%m)
        elif choice == 2:
            plt.plot(np.array(n_step_all)*dt,gamma_all, marker='o',label='$m=%i$'%m)
    
    title = "$m=%i$, $L=%i$, $dt=%.4f$"%(m, L, dt)
    plt.legend(loc='best')
    plt.title(title)
    plt.tight_layout()

    if choice == 0:
        plt.xlabel('$T$')
        plt.ylabel("$F$")     
    if choice == 1:
        plt.ylabel("\# of fid. eval")
        plt.xlabel("$T$")
    if choice == 2:
        plt.ylabel("$\Gamma$")
        plt.xlabel("$T$")
        
    plt.show()
        
    


    
    
    
    
    exit()
    #with open("long-ass-SA.pkl",'rb') as f:
    #    [fid_best, hx_tmp, fid_best_list, hx_tmp_list]=pickle.load(f)
    #symmetrize(hx_tmp,left_unchanged=True)
    #print(custom_prot.evaluate_protocol_fidelity(hx
    # _tmp))
    #plotting.protocol(np.arange(0.,250)*dt, hx_tmp)
    #exit()
    #print(gamma(hx_tmp))
    #exit()
    #hx_tmp=[-4,4,-4,4,4,4]
    #print(custom_prot.evaluate_protocol_fidelity(hx_tmp))
    #exit()

    '''for dt_ in np.arange(0.2,4.0,0.2)/20.:
        print("--------> T = ",dt_*20.)
        dt = dt_
        custom_prot=LZ.custom_protocol(
            J=-1.0,
            L=L, hz=1.0, hx_init_state=-2.0, hx_target_state=2.0,
            delta_t=dt, hx_i=-4., hx_max=4., action_set_=[-8.,0.,8.],
            option='fast'
        )

        fid=np.ones(2**20,dtype=np.float)
        for s in range(2**20):
            if s % 10000 == 0:
                print(s)
            hx_tmp=b2(s,w=20)
            fid[s]=custom_prot.evaluate_protocol_fidelity(hx_tmp)
        with open('fid_exact_T-%.3f_L-6_nstep-20.pkl'%(dt_*20),'wb') as f:
            pickle.dump(fid,f)

    exit()'''
    #np.histogram(fid,bins=100)
    #plt.show()
    #exit()
    #print(np.array(list(np.binary_repr(3, width=20)),dtype=np.int))
    #exit()
    #for a in range(tot):

    #x = (250 - 2*(t1+t2)) //2
    #print(2*x+2*t1+2*t2)
    '''for t1 in range(5,50):
        for t2 in range(5,50):
            x = (250 - 2*(t1+t2)) //2
            if (2*x+2*t1+2*t2 == 250) & (x > -1):
                rad = [[-4,4][np.random.randint(2)] for i in range(2*x)]
                hx_tmp=np.array([4]*t1+[-4]*t2 + rad +[4]*t2+[-4]*t1)
                #print(2*x+2*t1+2*t2,hx_tmp.shape,'\t',t1,t2)
                #print(hx_tmp.shape)
                fid = custom_prot.evaluate_protocol_fidelity(hx_tmp)
                if fid > fid_best:
                    fid_best = fid
                    hx_best = hx_tmp
                    t=[t1,t2]'''

    #print(custom_prot.evaluate_protocol_fidelity(hx_tmp))
    #exit()
    #np.random.seed(0)
    #fid_best, hx_tmp, fid_best_list = SD_symmetrized(n_step, custom_prot, n_refuse_max=10000, n_eval_max = 10000, init_random = True)
    #fid_best, 
    # hx_tmp, fid_best_list = SD_symmetrized(n_step, custom_prot, n_refuse_max=10000, n_eval_max = 200000, init_random = True)
    n_step = 320
    m = 0
    Ti=1e-3
    N_quench = 20000
    fid_best, hx_tmp, fid_best_list, hx_tmp_list = SA(Ti, N_quench, n_step, m, custom_prot, n_refuse_max = 20000, n_eval_max = 20000, init_random = True)
    
    fid_best_list=np.array(fid_best_list)
    plt.scatter(fid_best_list[:,0],fid_best_list[:,1])
    plt.show()

    Gamma=gamma(hx_tmp)
    plotting.protocol(np.arange(0.,n_step)*dt, hx_tmp, 
        title = "$m=%i$, $\Gamma=%.4f$, $F=%.12f$,\n $L=%i$, $dt=%.4f$, $T=%.3f$"%(m,Gamma,fid_best,L,dt,dt*n_step))
    

    exit()
    #with open("data/nstep-%i_m-0_L-6_dt-0p0100.pkl"%n_step,'wb') as f:
    #        pickle.dump([fid_best, hx_tmp, fid_best_list, hx_tmp_list],f)
    #        f.close()
    for n_step in range(10,400,10):
        fid_list = []
        fid_list_2 = []
        hx_tmp_list = []
        hx_tmp_list_2 = []
        
        m = 0
        Ti=1e-3
        N_quench = 20000
        fid_best, hx_tmp, fid_best_list, hx_tmp_list = SA(Ti, N_quench, n_step, m, custom_prot, n_refuse_max = 20000, n_eval_max = 20000, init_random = True)
        with open("data/nstep-%i_m-0_L-6_dt-0p0100.pkl"%n_step,'wb') as f:
            pickle.dump([fid_best, hx_tmp, fid_best_list, hx_tmp_list],f)
            f.close()
    exit()
        #fid_best, hx_tmp, fid_best_list, hx_tmp_list = SD(n_step, m, custom_prot, n_refuse_max=50000, n_eval_max = 400000, init_random = True)
        #print("Obtained fid: \t %.14f"%custom_prot.evaluate_protocol_fidelity(hx_tmp))
        #Gamma = gamma(hx_tmp)
        #fid_test = custom_prot.evaluate_protocol_fidelity(hx_tmp)
        #plotting.protocol(np.arange(0.,n_step)*dt, hx_tmp, 
        #title = "$m=%i$, $\Gamma=%.4f$, $F=%.12f$,\n $L=%i$, $dt=%.4f$, $T=%.3f$"%(m,Gamma,fid_test,L,dt,dt*n_step),
        #show = True
        #)

    '''with open("long-ass-SA.pkl",'wb') as f:
        pickle.dump([fid_best, hx_tmp, fid_best_list, hx_tmp_list],f)

    plotting.protocol_ising_2D_map(hx_tmp_list)

    fid_best_list=np.array(fid_best_list)
    plt.scatter(fid_best_list[:,0],fid_best_list[:,1])
    plt.show()
    exit()

    
    hx_tmp_cp=np.copy(hx_tmp)
    symmetrize(hx_tmp,left_unchanged=True)
    fid_symm_1=custom_prot.evaluate_protocol_fidelity(hx_tmp)
    print(fid_symm_1)
    symmetrize(hx_tmp_cp,left_unchanged=False)
    fid_symm_2=custom_prot.evaluate_protocol_fidelity(hx_tmp_cp)
    print(fid_symm_2)
    exit()'''


    '''m = 2
    for i in range(10):
        fid_best, hx_tmp, fid_best_list = SD(n_step, m, custom_prot, n_refuse_max=10000, n_eval_max = 200000, init_random = True)
        fid_list_2.append(fid_best)
        hx_tmp_list_2.append(fid_best)
    '''
    '''print("fid",fid_list)
    print("fid 2",fid_list_2)
    with open("out_fid.pkl","wb") as f:
        pickle.dump([fid_list,hx_tmp_list,fid_list_2,hx_tmp_list_2])
    '''
    #for i in range(10):
    #    fid_best, hx_tmp, fid_best_list = SD_symmetrized(n_step, custom_prot, n_refuse_max=10000, n_eval_max = 100000, init_random = True)
    
    #plotting.protocol(np.arange(0.0,n_step)*dt, hx_tmp, show=True)


    '''exit()
    for n_step in range(20,1601,20):
        SD_symmetrized(n_step, custom_prot, n_refuse_max=10000, n_eval_max = 200000, init_random = True)
    '''

def gamma(hx_tmp):
    n_step = len(hx_tmp)
    prod = hx_tmp[:n_step // 2]*(hx_tmp[n_step // 2:][::-1])
    pup = len(prod[prod > 0])
    return (1.*pup) / n_step

def swap(hx,x1,x2):
    tmp = hx[x2]
    hx[x2] = hx[x1]
    hx[x1] = tmp

def SD(N_step, sector, custom_prot_obj, n_refuse_max=100, n_eval_max = 10000, init_random = True, init_state=None, max_depth = 1000):
    """
    Stochastic descent in a particular magnetization sector
    """
    check_exhaus = True
    system = custom_prot_obj
    # perform random permutation
    n_up = int( (N_step + sector) / 2)
    fid_best = -10.

    hx_tmp = np.array([4]*n_up + [-4]*(N_step-n_up)) # initialize sector

    if init_random is True:
        np.random.shuffle(hx_tmp)
    if init_state is not None:
        hx_tmp = init_state

    fid_old = system.evaluate_protocol_fidelity(hx_tmp)
    n_refuse = 0
    n_tot_eval = 0
    fid_best_list=[]
    hx_tmp_list=[]
    fid_best = system.evaluate_protocol_fidelity(hx_tmp)
    hx_tmp_best = np.copy(hx_tmp)

    if check_exhaus is False:
        print("-> Stochastic descent !")
        while True:

            x1 = np.random.randint(N_step)
            x2 = np.random.randint(N_step)
            swap(hx_tmp,x1,x2)

            fid_new = system.evaluate_protocol_fidelity(hx_tmp)
            n_tot_eval += 1
            if n_tot_eval > n_eval_max:
                break
            if n_tot_eval % 1000 == 0:
                hx_tmp_list.append(hx_tmp_best)
                print(n_tot_eval,'\t\t',"%.14f"%fid_best,'\t',"%.5f"%gamma(hx_tmp_best))

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

    if check_exhaus is True:
        #g_old = 1.0
        print("--> Starting Stochastic Descent !")
        print("{0:<6}{1:<9}{2:<20}{3:<10}".format("#","n_eval","Fidelity","Gamma"))
        is_minima = True
        x1_ar, x2_ar = np.triu_indices(N_step,1)
        order = np.arange(0,x1_ar.shape[0],dtype=np.int)

        for _ in range(max_depth):
            np.random.shuffle(order)
            for pos in order:
                x1, x2 = (x1_ar[pos], x2_ar[pos])

                swap(hx_tmp,x1,x2)
                fid_new = system.evaluate_protocol_fidelity(hx_tmp)
                n_tot_eval +=1
                #g_new = gamma(hx_tmp)

                if (fid_new > fid_best) : #& (g_new < g_old):
                    #g_old = g_new
                    fid_best_list.append([n_tot_eval,fid_new])
                    hx_tmp_list.append(hx_tmp_best)
                    fid_best = fid_new
                    
                    swap(hx_tmp,x1,x2)
                    g1 = 0 if hx_tmp[x1]*hx_tmp[-(x1+1)]/16. < 0 else 1
                    g2 = 0 if hx_tmp[x2]*hx_tmp[-(x2+1)]/16. < 0 else 1
                    swap(hx_tmp,x1,x2)
                    out = "{0:<6}{1:<9}{2:<20.14f}{3:<10.5f}{4:10}{5:10}".format(_,n_tot_eval,fid_best, gamma(hx_tmp_best),g1,g2)
                    print(out)
                    hx_tmp_best = np.copy(hx_tmp)
                    is_minima = False
                    break
                swap(hx_tmp,x1,x2)

            if is_minima :
                print("minima reached !")
                print(_,'\t',n_tot_eval,'\t\t',"%.14f"%fid_best,'\t',"%.5f"%gamma(hx_tmp_best))
                break
            else:
                is_minima = True

    return fid_best, hx_tmp_best, fid_best_list, hx_tmp_list

def SA(Ti, N_quench, N_step, sector, custom_prot_obj, n_refuse_max=100, n_eval_max = 10000, init_random = True):
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
            #print(n_tot_eval,'\t\t',"%.14f"%fid_best,'\t',"%.5f"%gamma(hx_tmp))
        if n_tot_eval > n_eval_max:
            break
        if n_refuse > n_refuse_max: # number of random permutations refused before considering this is a local minima !
            break

    print('\n\n')
    fid_best, hx_tmp_best, fid_best_list, hx_tmp_list= SD(N_step, sector, custom_prot_obj, n_refuse_max=n_refuse_max, n_eval_max = n_eval_max,init_state=hx_tmp_best)
    
    return fid_best, hx_tmp_best, fid_best_list, hx_tmp_list


############ -> Symmetrized sector !  ---------------------------------##################################

def symmetrize(hx,left_unchanged=True):
    n_step_half = len(hx) // 2
    if left_unchanged is True:
        for i in range(n_step_half) :
            hx[-(i+1)] = - hx[i]
    else:
        for i in range(n_step_half) :
            hx[i] = -hx[-(i+1)]

def swap_symmetrize(hx,x1,x2):    
    tmp = hx[x2]
    hx[x2] = hx[x1]
    hx[x1] = tmp

    # resymmetrize !
    hx[-(x2+1)] = - hx[x2]
    hx[-(x1+1)] = - hx[x1]

def swap_symmetrize_2(hx,x1,x2):
    hx[x1] *= -1
    hx[-(x1+1)] = - hx[x1]


def SD_symmetrized(n_step, custom_prot_obj, n_refuse_max=100, n_eval_max = 10000, init_random = True):
    """
    Stochastic descent in a symmetric sector -> implies that m = 0 here 
    """
    assert n_step % 2 ==0, "Need an even number of steps !"

    system = custom_prot_obj
    n_half = n_step // 2  
    if init_random is True:
        hx_tmp = np.random.choice([-4,4],size=n_step)
    else:
        hx_tmp=np.array([4]*n_half + [-4]*n_half) # initialize with ONE bang
    
    symmetrize(hx_tmp) # symmetrizes the state
    assert np.sum(hx_tmp) < 1.e-6, "Error -> should be zero !"

    fid_old = system.evaluate_protocol_fidelity(hx_tmp)
    fid_best = -10.
    n_refuse = 0
    n_tot_eval = 0
    fid_best_list=[]

    while True:

        x1 = np.random.randint(n_half)
        x2 = np.random.randint(n_half)
        swap_symmetrize_2(hx_tmp,x1,x2)
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
            swap_symmetrize_2(hx_tmp,x1,x2) # swap back
            n_refuse +=1
        
        if n_refuse > n_refuse_max: # number of random permutations refused before considering this is a local minima !
            break

    return fid_best, hx_tmp, fid_best_list


if __name__ == "__main__" :
    main()