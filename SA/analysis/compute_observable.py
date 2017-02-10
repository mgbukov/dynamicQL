'''
Created on Jan 3 , 2017

@author: Alexandre Day


Purpose:

Computes various observables for the LZ problem

'''

import sys
import numpy as np
from quspin.tools.measurements import ent_entropy

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
    OP1,OP2=0.,0.
    h_protocol=h_protocol/np.max(normalize_factor) # So protocol is normalized (between -1 and 1)
    
    for t in range(n_time):
        ht=h_protocol[:,t]
        s=np.sum(ht)
        OP1+=np.dot(ht,ht)
        OP2+=s*s
        #print(t,OP1,OP2)
    
    OP=OP1/n_sample-OP2/(n_sample*n_sample)
    return OP/(n_time)


def Average_Overlap(h_protocol,normalize_factor):
    """
    Purpose:
        Computes the average overlap between protocols
        
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

def MB_observables(protocol, param_SA, matrix_dict, fin_vals=False):

    global hx_discrete
    """
    this function returns instantaneous observalbes during ramp 
    OR 
    when fin_vals=True only the final values at the end of the ramp
    ----------------
    observables:
    ----------------
    Fidelity
    E: energy above instantaneous GS
    delta_E: energy fluctuations
    Sd: diagonal entropy
    Sent: entanglement entropy
    """

    # calculate final basis
    V_target=param_SA['V_target']
    hx_i=param_SA['hx_initial_state']
    hx_f=param_SA['hx_final_state']
    J=param_SA['J']
    L=param_SA['L']
    hz=param_SA['hz']
    H=param_SA['H']

    # calculate initial state
    psi=param_SA['psi_i']
    psif=param_SA['psi_target'].squeeze()


    psi=psi.squeeze()
    # define Sent subsystem
    subsys=[i for i in range(L//2)]

    # preallocate variables
    Fidelity,E,delta_E,Sd,Sent=([],[],[],[],[])

    i=0
    hx_discrete=protocol
    while True:
        # instantaneous fidelity
        Fidelity.append( abs(psi.conj().dot(V_target[:,0]))**2 )
        # excess energy above inst energy
        EGS = H.eigsh(k=1, which='SA', maxiter=1E10, return_eigenvectors=False)
        E.append( H.matrix_ele(psi,psi).real/L - EGS/L )
        # inst energy density
        delta_E.append( np.sqrt( (H(time=0)*H).matrix_ele(psi,psi) - H.matrix_ele(psi,psi)**2).real/L )
        # diagonal entropy in target basis
        pn = abs( V_target.conj().T.dot(psi) )**2.0 + np.finfo(psi[0].dtype).eps
        Sd.append( -pn.dot(np.log(pn))/L)
        # entanglement entropy
        Sent.append( ent_entropy(psi,H.basis,chain_subsys=subsys)['Sent'] )
        
        if i == len(protocol)-1:
            break
        else:
            # go to next step
            b=hx_discrete[i] # --> induces a change in H
            psi=matrix_dict[b].dot(psi)
            i+=1
            
    if fin_vals:
        return Fidelity[-1],E[-1],delta_E[-1],Sd[-1],Sent[-1]
    else:
        return Fidelity,E,delta_E,Sd,Sent


if __name__=="__main__":
    main()