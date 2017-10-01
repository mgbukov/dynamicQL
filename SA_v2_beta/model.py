import numpy as np
import scipy.linalg as sla
import time
import copy
from functools import reduce


class MODEL:
    def __init__(self, H, param):
        
        self.H = H  # shallow copy 
        self.param = param

        # calculate initial and final states wave functions (ground-state)
        if self.param['norm']=='GS':
            self.psi_i = H.ground_state(hx=param['hx_i']).squeeze()
            self.psi_target = H.ground_state(hx=param['hx_f']).squeeze()
            self.norm=1.0
        elif self.param['norm']=='trace':
            self.psi_i = H.eigen_basis(hx=param['hx_i']) # initial eigenbasis
            self.psi_target = H.eigen_basis(hx=param['hx_f'])
            self.norm=self.H.basis.Ns
        
        self.H_target = H.evaluate_H_at_hx(hx=self.param['hx_f']).toarray()
        
        print( "Initial overlap is \t %.5f" % self.overlap(self.psi_i, self.psi_target))
    
        self.dt=self.param['dt']
        self.n_h_field = len(self.H.h_set)
        self.precompute_mat = {} # precomputed evolution matrices are indexed by integers 
    
        print("\nPrecomputing evolution matrices ...")
        start=time.time()
        self.precompute_expmatrix()
        print("Done in %.4f seconds"%(time.time()-start))


        if param['task']=='GRAPE':
            if param['L'] == 1:
                print("\nDefining evolution operator ...")
                self._precompute_analytic_evo_op_L1()
                print("Done in %.4f seconds"%(time.time()-start))
            elif param['L'] == 2:
                print("\nDefining evolution operator ...")
                self._precompute_analytic_evo_op_L2()
                print("Done in %.4f seconds"%(time.time()-start))


    def overlap(self,psi1,psi2):
        # Square of overlap between two states
        '''
        if self.param['norm']=='GS':
            t=(np.abs(np.dot(psi1.T.conj(),psi2))**2).squeeze()
        elif self.param['norm']=='trace':
            t=1.0/self.H.basis.Ns*np.sum(np.abs(np.einsum('ij,ij->j',psi1.conj(),psi2))**2)
            #t=1.0/self.H.basis.Ns*np.einsum('ij,ij,kj,kj->',psi1.conj(),psi2,psi1,psi2.conj()).real
            #t=1.0/self.H.basis.Ns*np.diag( np.abs(np.dot(psi1.T.conj(),psi2))**2 ).sum()
        '''

        t=1.0/self.norm*np.sum(np.abs(np.einsum('i...,i...->...',psi1.conj(),psi2))**2)

        return t


    def precompute_expmatrix(self):
        # Precomputes the evolution matrix and stores them in a dictionary
        
        # set of possible -> fields 
        h_set = self.H.h_set

        for idx, h in zip(range(len(h_set)),h_set):
            self.precompute_mat[idx] = sla.expm(-1j*self.dt*self.H.evaluate_H_at_hx(hx=h).toarray())
        self.param['V_target'] = self.H.eigen_basis(hx=self.param['hx_f'])
    
    def compute_evolved_state(self, protocol=None, discrete=True): 
        # Compute the evolved state after applying protocol
        if protocol is None:
            protocol = self.H.hx_discrete
            discrete=True

        psi_evolve=self.psi_i.copy()
        
        if discrete:
            for idx in protocol:
                psi_evolve = self.precompute_mat[idx].dot(psi_evolve)
        else:
            for h in protocol:
                psi_evolve = sla.expm(-1j*self.dt*self.H.evaluate_H_at_hx(hx=h).toarray()).dot(psi_evolve) 

        return psi_evolve

    def compute_protocol_gradient(self,protocol=None):
        """This function follows the GRAPE algorithm in the Supplemental material of arXiv:1705.00565.

        """
        if protocol is None:
            protocol = self.H.hx_discrete
 
        psi=np.zeros(self.psi_i.shape+protocol.shape,dtype=np.complex128 )
        phi=np.zeros_like(psi)

        psi_evolve=self.psi_i.copy() # |ket> state
        phi_evolve=self.psi_target.copy().T.conj() # <bra| state

        
        for i, (h,h_rev) in enumerate(zip(protocol, protocol[::-1])):
            # evolve psi forward: U(t,0)|psi>
            psi[:,i] = psi_evolve
            #psi_evolve = sla.expm(-1j*self.dt*self.H.evaluate_H_at_hx(hx=h).toarray()).dot(psi_evolve)
            psi_evolve = self.analytic_evolution_op(hx=h).dot(psi_evolve)
            
            #print(sla.expm(-1j*self.dt*self.H.evaluate_H_at_hx(hx=h).toarray())  -  self.analytic_evolution_op(hx=h))
            #print()
            #exit()

            # evolve phi backward: <phi|U(T,t)
            #phi_evolve = phi_evolve.dot( sla.expm(-1j*self.dt*self.H.evaluate_H_at_hx(hx=h_rev).toarray()) )
            phi_evolve = phi_evolve.dot( self.analytic_evolution_op(hx=h_rev) )
            
            phi[:,i]=phi_evolve

        # compute overlap and prefactor
        prefactor=psi_evolve.conj().dot(self.psi_target)
        double_overlap=np.einsum('ij,ik,kj->j',phi[:,::-1],self.H.control_hamiltonian.toarray(),psi)
        
        protocol_gradient = 2.0*np.imag( prefactor*double_overlap )
        
        #protocol_gradient = 2.0/self.norm*np.sum(np.imag( prefactor*np.array(double_overlap) ),axis=1)
        
        return protocol_gradient


    def compute_fidelity(self, protocol = None, psi_evolve = None, discrete=True):
        # Computes fidelity for a given protocol
        if psi_evolve is None:
            psi_evolve = self.compute_evolved_state(protocol=protocol,discrete=discrete)

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

    def _precompute_analytic_evo_op_L2(self):
        """ Precomputes helper functions for analytic_evolution_op() for L=2

        """

        p = lambda x: 12.0*(3.0*(x**2 + self.param['hz']**2) + self.param['J']**2)
        q = lambda x: -8.0j*self.param['J']*(9.0*x**2 + 2.0*(-9.0*self.param['hz']**2 + self.param['J']**2))
        S = lambda p,q: -2.0*np.lib.scimath.sqrt(-p/3.0)*np.cos(1.0/3.0*np.arccos(3.0*q/(2.0*p)*np.lib.scimath.sqrt(-3.0/p) ) )
        # roots of cubic eqn
        t0 = lambda x: S(p(x),q(x))
        t2 = lambda x:-S(p(x),-q(x))
        #t1 = lambda x: -t0(x)-t2(x)
        
        # define matrix exponential
        denom = lambda n, hx: 12.0*hx**2 + 12.0*self.param['hz']**2 + 4.0*self.param['J']**2 + n**2
        
        U11 = lambda n, hx: 1.0/3.0*(18.0*np.exp(self.dt*n/6.0)*hx**2 + 24.0*np.exp(self.dt*n/6.0)*self.param['hz']*self.param['J'] + 8.0*np.exp(self.dt*n/6.0)*self.param['J']**2 - 6.0j*np.exp(self.dt*n/6.0)*self.param['hz']*n + 2.0j*np.exp(self.dt*n/6.0)*self.param['J']*n + np.exp(self.dt*n/6.0)*n**2)
        U12 = lambda n, hx: np.sqrt(2)*hx*(6.0*np.exp(self.dt*n/6.0)*self.param['hz'] + 2.0*np.exp(self.dt*n/6.0)*self.param['J'] + 1j*np.exp(self.dt*n/6.0)*n )
        U13 = lambda n, hx: -6.0*hx**2*np.exp(self.dt*n/6.0)
        
        U22 = lambda n, hx: 1.0/3.0*( 36.0*np.exp(self.dt*n/6.0)*self.param['hz']**2 - 4.0*np.exp(self.dt*n/6.0)*self.param['J']**2 - 4.0j*np.exp(self.dt*n/6.0)*self.param['J']*n + np.exp(self.dt*n/6.0)*n**2 )
        U23 = lambda n, hx: np.sqrt(2)*hx*(-6.0*np.exp(self.dt*n/6.0)*self.param['hz'] + 2.0*np.exp(self.dt*n/6.0)*self.param['J'] + 1j*np.exp(self.dt*n/6.0)*n )
        
        U33 = lambda n, hx: 1.0/3.0*( 18.0*np.exp(self.dt*n/6.0)*hx**2 - 24.0*np.exp(self.dt*n/6.0)*self.param['hz']*self.param['J'] + 8.0*np.exp(self.dt*n/6.0)*self.param['J']**2 + 6j*np.exp(self.dt*n/6.0)*self.param['hz']*n + 2j*np.exp(self.dt*n/6.0)*self.param['J']*n + np.exp(self.dt*n/6.0)*n**2 )

        # sum over cubit roots to get final evolution operator as a function of hx
        self.analytic_evolution_op = lambda hx: np.sum(   np.exp(1j*self.dt*self.param['J']/6.0)/denom(n, hx)*np.array( 
                                                [ [U11(n, hx),U12(n, hx),U13(n, hx)],
                                                  [U12(n, hx),U22(n, hx),U23(n, hx)],
                                                  [U13(n, hx),U23(n, hx),U33(n, hx)] ]
                                            )
                                        for n in [t0(hx),-t0(hx)-t2(hx),t2(hx)]
                                        )


    def _precompute_analytic_evo_op_L1(self):
        """ Precomputes helper functions for analytic_evolution_op() for L=1

        """

        sr = lambda x: np.sqrt(x**2 + self.param['hz']**2)

        U11 = lambda x: np.cos(0.5*self.dt*sr(x)) - 1j*self.param['hz']/sr(x)*np.sin(0.5*self.dt*sr(x))
        U12 = lambda x: 1j*x/sr(x)*np.sin(0.5*self.dt*sr(x))

        self.analytic_evolution_op = lambda hx:  np.array( [[U11(hx),U12(hx)],
                                                             [U12(hx),U11(hx).conj()]] )


            

