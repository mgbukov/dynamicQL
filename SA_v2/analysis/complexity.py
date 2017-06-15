import numpy as np
from matplotlib import pyplot as plt
import pickle

#measure size of connected region ... this gives you some kind of measure ...

def main():
    f=open('../scaling_SD2.pkl','rb')
    data=pickle.load(f)

    CvsT={}
    plt.plot(data[round(2.0,2)][:,0],np.log(data[round(2.0,2)][:,1]))
    plt.show()
    exit()
    #for t in np.arange(0.1,4.01,0.1):
    #    plt.plot(data[round(t,2)][:,0],np.log(data[round(t,2)][:,1]))
    #plt.show()
    #exit()
    #print(data[3.5])
    #exit()
    r=range(4,29,2)
    for n,n_idx in zip(r,range(len(r))):
        current_c = []
        for t in np.arange(0.1,4.01,0.1):
            current_c.append(data[round(t,2)][n_idx,1])
        CvsT[n]=np.copy(current_c)
    
    for n in range(4,29,2):
        plt.plot(np.arange(0.1,4.01,0.1),CvsT[n])
    
    plt.show()
    #print(CvsT[4])
    #print(CvsT[4].shape)
    exit()
    
if __name__ == '__main__':
    main()