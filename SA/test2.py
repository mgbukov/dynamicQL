import numpy as np
from matplotlib import pyplot as plt


a=np.array([1,2,3])
plt.scatter([1,2,3],a)
a=a*2
plt.scatter([1,2,3],a)
plt.show()


#print(np.bincount([1,-1,-1,-1,1]))

