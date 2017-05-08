import numpy as np
import time

def overlap(psi1,psi2):
    # Square of overlap between two states
    return abs(np.sum(np.conj(psi1)*psi2))**2

class MODEL:
    def __init__(self, HvsT, param):
        
        self.HvsT = HvsT
        self.param = param

        # calculate initial and final states wave functions (ground-state)
        self.psi_i = HvsT.evaluate_ground_state_hx_special(hx=param['hx_i'])
        self.psi_target = HvsT.evaluate_ground_state_hx_special(hx=param['hx_f'])
        
        print( "Initial overlap is \t %.5f" % overlap(self.psi_i,self.psi_target) )
    
        self.param = param
        self.psi_evolve = None
        self.precompute_mat = {}
    
        print("\nPrecomputing evolution matrices ...")
        start=time.time()
        self.precompute_expmatrix()
        print("Done in %.4f seconds"%(time.time()-start))

    def precompute_expmatrix(self):
    
        """
        Purpose:
            Precomputes the evolution matrix and stores them in a dictionary
        """
        from scipy.linalg import expm
        
        # set of possible -> fields 
        h_set = self.HvsT.h_set
        for idx, h in zip(range(len(h_set)),h_set):
            init_h = self.HvsT.hx_discrete[0]
            self.HvsT.update_hx_real(time=0, hx=h)
            self.precompute_mat[idx] = expm(-1j*self.param['dt']*self.HvsT.hamiltonian.todense())
            self.HvsT.update_hx(time=0, hx_idx=init_h)
        
        init_h = self.HvsT.hx_discrete[0]
        self.HvsT.update_hx_real(time=0, hx=self.param['hx_f'])
        self.param['V_target'] = self.HvsT.eigen_basis()
        self.HvsT.update_hx(time=0, hx_idx=init_h)
    
    def compute_fidelity(self, protocol=None):
        if protocol is None:
            protocol = self.HvsT.hx_discrete
        psi_evolve=self.psi_i.copy()

        for idx in protocol:
             psi_evolve = self.precompute_mat[idx].dot(psi_evolve)
        self.psi_evolve = psi_evolve.copy()

        return abs(np.sum(np.conj(psi_evolve)*self.psi_target))**2
    def update_protocol(self, protocol):
        self.HvsT.hx_discrete=protocol
    def protocol(self):
        return self.HvsT.hx_discrete

    def compute_energy(self,psi = None):
        H = self.HvsT.evaluate_H_hx_special(hx=self.param['hx_f'])
        if psi is None:
            if self.psi_evolve is None:
                self.compute_fidelity() # computes final state w.r.t to current protocol
            return np.real(np.dot(np.conj(self.psi_evolve).T,np.dot(H,self.psi_evolve))[0,0])
        else:
            return np.real(np.dot(np.conj(psi).T,np.dot(H,psi))[0,0])
