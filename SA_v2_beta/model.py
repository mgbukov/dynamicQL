import numpy as np
import time
import copy
from functools import reduce


class MODEL:
    def __init__(self, H, param):
        
        self.H = H  # shallow copy 
        self.param = param

        # calculate initial and final states wave functions (ground-state)
        if self.param['norm']=='GS':
            self.psi_i = H.ground_state(hx=param['hx_i'])
            self.psi_target = H.ground_state(hx=param['hx_f'])
        elif self.param['norm']=='trace':
            self.psi_i = H.eigen_basis(hx=param['hx_i']) # initial eigenbasis
            self.psi_target = H.eigen_basis(hx=param['hx_f'])
        
        self.H_target = H.evaluate_H_at_hx(hx=self.param['hx_f']).toarray()
        
        print( "Initial overlap is \t %.5f" % self.overlap(self.psi_i, self.psi_target))
    
        self.n_h_field = len(self.H.h_set)
        self.precompute_mat = {} # precomputed evolution matrices are indexed by integers 
    
        print("\nPrecomputing evolution matrices ...")
        start=time.time()
        self.precompute_expmatrix()
        print("Done in %.4f seconds"%(time.time()-start))

    def overlap(self,psi1,psi2):
        # Square of overlap between two states
        if self.param['norm']=='GS':
            t=(np.abs(np.dot(psi1.T.conj(),psi2))**2).squeeze()
        elif self.param['norm']=='trace':
            t=1.0/self.H.basis.Ns*np.sum(np.abs(np.einsum('ij,ij->j',psi1.conj(),psi2))**2)
            #t=1.0/self.H.basis.Ns*np.einsum('ij,ij,kj,kj->',psi1.conj(),psi2,psi1,psi2.conj()).real
            #t=1.0/self.H.basis.Ns*np.diag( np.abs(np.dot(psi1.T.conj(),psi2))**2 ).sum()

        return t


    def precompute_expmatrix(self):
        # Precomputes the evolution matrix and stores them in a dictionary
        from scipy.linalg import expm
        
        # set of possible -> fields 
        h_set = self.H.h_set

        for idx, h in zip(range(len(h_set)),h_set):
            self.precompute_mat[idx] = expm(-1j*self.param['dt']*self.H.evaluate_H_at_hx(hx=h).toarray())

        self.param['V_target'] = self.H.eigen_basis(hx=self.param['hx_f'])
    
    def compute_evolved_state(self, protocol=None): 
        # Compute the evolved state after applying protocol
        if protocol is None:
            protocol = self.H.hx_discrete
        psi_evolve=self.psi_i.copy()
        
        for idx in protocol:
            psi_evolve = self.precompute_mat[idx].dot(psi_evolve)
        return psi_evolve

    def compute_fidelity(self, protocol = None, psi_evolve = None):
        # Computes fidelity for a given protocol
        if psi_evolve is None:
            psi_evolve = self.compute_evolved_state(protocol=protocol)

        return self.overlap(psi_evolve, self.psi_target)

    def update_protocol(self, protocol, format = 'standard'):
        # Update full protocol
        # Format is integers or real values (which are then converted to integers)

        if format is 'standard':
            self.H.hx_discrete = copy.deepcopy(np.array(protocol))
        elif format is 'real':
            n_bang = len(protocol)
            for t, h in zip(range(n_bang),protocol):
                self.H.update_hx(time=t,hx=h) 
        self.psi_evolve = None

    def update_hx(self, time:int, hx_idx:int):
        # Update protocol at a specific time
        self.H.hx_discrete[time]=hx_idx

    def protocol(self):
        # Returns current protocol
        return self.H.hx_discrete
    
    def protocol_hx(self, time):
        # Returns protocol value at time
        return self.H.hx_discrete[time]
    
    def random_flip(self,time):
        # Proposes a random update a time :
        current_h = self.H.hx_discrete[time]
        return np.random.choice(list(set(range(self.n_h_field))-set([current_h])))

    def swap(self, time_1, time_2):
        tmp = self.H.hx_discrete[time_1]
        self.H.hx_discrete[time_1] = self.H.hx_discrete[time_2]
        self.H.hx_discrete[time_2] = tmp

    def compute_Sent(self,protocol=None,psi_evolve = None):
        # Compute Sent of evolved state acc. to Hamiltonian
        if psi_evolve is None:
            psi_evolve = self.compute_evolved_state(protocol=protocol)
        return self.H.basis.ent_entropy(psi_evolve.squeeze(),enforce_pure=True)['Sent_A']

    def compute_energy(self, protocol = None, psi_evolve = None):
        # Compute energy of evolved state w.r.t to target Hamiltonian

        if psi_evolve is None:
            psi_evolve = self.compute_evolved_state(protocol=protocol)
        return np.diag((np.dot(psi_evolve.T.conj(), np.dot(self.H_target, psi_evolve))).real) #[0,0]

    def compute_observables(self, protocol = None, psi_evolve = None):
        # Compute energy of evolved state w.r.t to target Hamiltonian

        if psi_evolve is None:
            psi_evolve = self.compute_evolved_state(protocol=protocol)

        pri_evolve=psi_evolve.squeeze()

        E = np.diag((np.dot(psi_evolve.T.conj(), np.dot(self.H_target, psi_evolve))).real) #[0,0]
        deltaE = (np.sqrt( 
                    np.diag( reduce(np.dot,[psi_evolve.T.conj(),(self.H.hamiltonian_discrete(time=0)*self.H_target),psi_evolve]) )
                  - np.diag( reduce(np.dot,[psi_evolve.conj().T,self.H_target,psi_evolve]) )**2
                         ).real/(self.H.basis.L)
                 ).squeeze()
        Sent = self.H.basis.ent_entropy(psi_evolve,enforce_pure=True)['Sent_A']
        
        return E, deltaE, Sent
    
    def compute_magnetization(self, protocol = None):
        if protocol is None:
            protocol = self.H.hx_discrete
        Nup = np.sum(protocol)
        Ndown = protocol.shape[0] - Nup
        return Nup - Ndown

