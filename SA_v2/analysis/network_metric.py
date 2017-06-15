import numpy as np
import sys
import pickle
sys.path.append("..")


file = "../data/ES_L=06_dt=0p0200_nStep=0020_T=0p40_nQuench=10000_Ti=m1p00_symm=0_J=1p00_hz=1p00_hxI=m2p00_hxF=2p00_hxmax=4p00_hxmin=m4p00_dh=8p00.pkl"

data = pickle.load(open(file,'rb'))

fid, energy = data[:,0], data[:,1]

def flip(h10,t):
    return h10^2**t

n_step = 20
#n_state = fid.shape[0]
#idx_gs = np.argmax(fid)

print(np.sum(connection))

def measure_connectivity(fid, n_step):
    n_state = fid.shape[0]
    idx_gs = np.argmax(fid)
    connection = -1*np.ones(n_state,dtype=np.int)
    for a in range(n_state):
        a1 = a
        path = [a1]
        while True: # till the faith is determined
            not_connected = True
            if connection[a1] != -1: # stopping condition, faith is determined ... move on
                break
            for t in range(n_step):
                a2=flip(a1,t)
                if connection[a2] == 1:
                    for a_prev in path:
                        connection[a_prev] = 1 # all encountered states are connected !
                    not_connected = False
                    break
                elif fid[a2] > fid[a1] and connection[a2] != 0:
                    # accept and move on
                    a1 = a2
                    not_connected = False
                    path.append(a2)
                    break

            if not_connected:
                connection[a1] = 0
                break