import numpy as np
import pickle
from matplotlib import pyplot as plt

#file=open("data/a1_t=30_fideval=500.pkl","rb")
#file=open("data/marin.pkl","rb")
#res1=pickle.load(file)
#print(res1)

data=[ 4.00000001,  4.00000001,  3.99999999,  3.99999999,  3.99999999,
        3.99999996,  0.6       ,  2.39999997,  2.70000003, -0.80000004,
       -1.39999998,  1.09999998, -3.3       , -1.3       , -4.00000004,
       -3.99999999, -4.00000003, -4.00000004, -4.00000004, -4.00000001]
plt.step(range(20),data,where='post')
plt.show()  