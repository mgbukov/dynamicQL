from plots import plotting
import pickle
import numpy as np
from scipy.stats import describe as descSCIPY

np.set_printoptions(suppress=True)

def main():
    Nstep=[50,100,150,170,200,250]
    fidelities={}
    descript=[]
    for n in Nstep:
        with open('data/DOS_fid-5000_dt-0p01_Nstep-%i.pkl'%n,'rb') as f:
            fidelities[n]=pickle.load(f);f.close();
            descript.append([n,n*0.01]+list(describe(fidelities[n])))
    
    print(np.array(descript))
    

def describe(X):
    desRE=descSCIPY(X)
    return desRE.minmax[0],desRE.minmax[1],desRE.mean,desRE.variance,desRE.skewness,np.percentile(X,25),np.percentile(X,75)
    #return descSCIPY(X),np.percentile(X,25),np.percentile(X,75)

if __name__=='__main__':
    main()