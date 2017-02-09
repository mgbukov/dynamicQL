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

L=16
hx_tmp=list([-4.,-4.,-4.,4.,4.,4.,4.,4.,-4.,4.,4.])
action_set=[-8.,0.,8.]

custom_prot=LZ.custom_protocol(
    L=L, hz=1.0, hx_init_state=-2.0, hx_target_state=2.0,
    delta_t=0.05,hx_i=-4.,hx_max=4.,action_set_=[-8.,0.,8.],
    option='fast'
)
print(custom_prot.evaluate_protocol_fidelity(hx_tmp))
#print("fast_eval",fast_eval)
exit()
start=time.time()
print(standard_eval.evaluate_protocol_fidelity(hx_tmp))
print("Standard run in %.6f"%(time.time()-start))

'''start=time.time()
print(fast_eval.evaluate_protocol_fidelity(hx_tmp))
print("Fast ran in %.6f"%(time.time()-start))
'''