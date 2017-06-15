import numpy as np
from tsne_visual import TSNE
from sklearn.decomposition import PCA
import sys
sys.path.append("..")
from utils import UTILS
import pickle
from matplotlib import pyplot as plt

utils=UTILS()
parameters = utils.read_parameter_file(file="../para.dat")
n_step = 20


parameters['T'] = round(0.4,2)
parameters['n_step'] = n_step
parameters['dt'] = parameters['T']/parameters['n_step']

file=utils.make_file_name(parameters,root='../data/')

n_state = 10000
b2_array = lambda n10 : np.array(list(np.binary_repr(n10, width=n_step)), dtype=np.float)

pipe = open(file,'rb')
data = pickle.load(pipe)
fid, ene = data[:,0], data[:,1]
f_state = np.argsort(fid)[-n_state:]

X = np.zeros((n_state,20),dtype=np.float)
for i in range(X.shape[0]):
    X[i] = b2_array(f_state[i])

tsne=TSNE(n_components=2)
#pca=PCA(n_components=2)
Xnew=tsne.fit_transform(X)
#Xnew=pca.fit_transform(X)
#print(pca.explained_variance_ratio_)
plt.scatter(Xnew[:,0],Xnew[:,1])
plt.show()
#print(X[:10])




#print(data.shape)




## load in exact spectrum
## -> run over spectrum
