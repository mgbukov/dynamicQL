import numpy as np
from matplotlib import pyplot as plt
import itertools

print(np.arange(10))
exit()
a=np.array([1,2,3])
b=np.array([5,6,7])
for i,j in itertools.product(a, b):
    print(i,j)
#print(itertools.product(a, b))
#print(np.random.randint((10 // 2)))
#print(np.bincount([1,-1,-1,-1,1])):

