'''
Created on Jan 3 , 2017

@author: Alexandre Day


Purpose:

Computes various observables for the LZ problem

'''

import sys
sys.path.append("..")
import numpy as np
from sklearn.manifold import TSNE


def main():
    
    print("hello")    
    
def Ed_Ad_OP(h_protocol,normalize_factor):
    """
    Purpose:
        Computes Edward-Anderson order parameter
        
    Parameter:
        Array of protocols obtained
    """
    assert len(h_protocol.shape) ==2, "Need an array of shape (n_sample,t_evolv)"
    assert h_protocol.shape[0]>4, "Need more than a few samples !"
    n_sample,n_time=h_protocol.shape
    OP=0.
    h_protocol=h_protocol/np.max(normalize_factor) # So protocol is normalized (between -1 and 1)
    
    for t in range(n_time):
        corr=0.
        for a in range(n_sample):
            for b in range(n_sample):
                corr+=h_protocol[a,t]*h_protocol[b,t]
        norm_t=np.linalg.norm(h_protocol[:,t])
        corr-=norm_t*norm_t
        OP+=corr
    return OP/(n_time*n_sample*n_sample)

def run_tsne(h_protocol):
    """
    Purpose:
        Computes a t-SNE embedding of the protocols using sklearn
    Parameter:
        Array of protocols obtained
    Return: array, shape (n_sample,2) 
    
    """
    h_protocol=h_protocol/np.max(h_protocol)
    
    assert len(h_protocol.shape) ==2, "Need an array of shape (n_sample,t_evolv)"
    assert h_protocol.shape[0]>10, "Need more than a few samples !!"
    
    tsne=TSNE(n_components=2, learning_rate=100.0, n_iter=1000, n_iter_without_progress=50, min_grad_norm=1e-07, metric='euclidean', 
              init='random', verbose=1, random_state=None, method='exact', angle=0.5)
    
    return tsne.fit_transform(h_protocol)
    
if __name__=="__main__":
    main()