'''
Created on Jan 3 , 2017

@author: Alexandre Day


Purpose:

Computes various observables for the LZ problem

'''

import sys
sys.path.append("..")
import numpy as np



def main():
    
    
    print("hello")    
    
def Ed_Ad_OP(h_protocol):
    """
    Purpose:
        Computes Edward-Anderson order parameter
    """
    assert len(h_protocol.shape) ==2, "Need an array of shape (n_sample,t_evolv)"
    assert h_protocol.shape[0]>4, "Need more than a few samples !"
    n_sample,n_time=h_protocol.shape
    OP=0.
    h_protocol=h_protocol/np.max(h_protocol) # So protocol is normalized (between -1 and 1)
    
    for t in range(n_time):
        corr=0.
        for a in range(n_sample):
            for b in range(n_sample):
                corr+=h_protocol[a,t]*h_protocol[b,t]
        norm_t=np.linalg.norm(h_protocol[:,t])
        corr-=norm_t*norm_t
        OP+=corr
    return OP/(n_time*n_sample*n_sample)


if __name__=="__main__":
    main()