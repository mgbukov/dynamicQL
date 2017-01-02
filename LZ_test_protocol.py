"""
Purpose:
    This module is just for testing protocols
"""

import LZ_sim_anneal as LZ

hx_tmp=[ 4.,4.,4.,4.,4.,4.,4.,-4.,4.,-4.,4.,-4.,4,-4.,-4.,-4.,-4.,-4.,-4.,-4.]
param_check={"J":1.236,"L":1,"hz":1.0,"hx_i":-1.0,"hx_f":1.0,"delta_t":0.05}
print(LZ.check_custom_protocol(hx_tmp,**param_check))