import copy
import numpy as np
import time 

#print(list(set(range(2))-set([0])))
s = time.time()
#np.random.choice([0,1])
np.random.randint(0,2)
print(np.random.choice(list(set(range(2))-set([1]))))
print(time.time() - s)


#print(np.random.randint(0,2))

'''
class T:
    def __init__(self):
        self.prot = [0,1,2#]
    def update(self,prot):
        self.prot = copy.deepcopy(prot)

a=T()
b=[-2,-2,-3,-4]
a.update(b)
print(a.prot)
b[0]=100
print(a.prot)
a.prot[0]=200
print(a.prot)'''