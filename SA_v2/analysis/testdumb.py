import numpy as np
import matplotlib
#print(matplotlib.matplotlib_fname())
#exit()
from matplotlib import pyplot as plt


plt.scatter(list(np.arange(0.01,0.1,0.01)),[1,2,3,4,5,6,7,8,9])
plt.show()


#x=np.array([[1,1,1],[1,0,1],[0,1,1],[1,1,1],[0,1,1]])
#xunique,count=np.unique(x, return_counts=True, axis=0)
#print(count)ls