'''
Created on Jan 2 , 2017

@author: Alexandre Day

Purpose:
    This module is just for testing protocols

'''

import LZ_sim_anneal as LZ
import time

hx_tmp=[-4.,4.,4.,-4.,4.,-4.]
param_check={"J":1.236,"L":1,"hz":1.0,"hx_init_state":-1.0,"hx_target_state":1.0,"delta_t":0.05}

#
start=time.time()
print(LZ.check_custom_protocol(hx_tmp,**param_check,option='standard'))
print("Standard run in %.4f"%(time.time()-start))

start=time.time()
print(LZ.check_custom_protocol(hx_tmp,**param_check,option='fast'))
print("Fast ran in %.4f"%(time.time()-start))
