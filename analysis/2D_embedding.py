'''
Created on Jan 14 , 2017

@author: Alexandre Day

'''
from sklearn.decomposition import PCA
import sys
from tsne.tsne import TSNE
sys.path.append("..")
import utilities as ut
import time
from plots import plotting
import pickle
import numpy as np
np.set_printoptions(suppress=True)
from matplotlib import pyplot as plt


def main():
    
    param=ut.default_parameters()
    
    param['N_time_step']=500
    param['delta_t']=0.0004
    start=time.time()
    data=ut.gather_data(param,'../data/')
    
    dt_list=np.arange(0.0002,0.0061,0.0002)
    #dt_list=np.arange(0.0002,0.0061,0.0004)
    n_element=len(dt_list)
    #print(dt_list)
    
    #===========================================================================
    # fid,std_fid,n_fid,h_prot,x,fid_all=([],[],[],[],[],[])
    # 
    # for dt in dt_list:
    #     param['delta_t']=dt
    #     data=ut.gather_data(param,'../data/')
    #     
    #     if data:
    #         n_fid.append(np.mean(data['n_fid']))
    #         fid.append(np.mean(data['fid']))
    #         fid_all.append(data['fid'])
    #         std_fid.append(np.std(data['fid']))
    #         h_prot.append(data['h_protocol'])
    #         x.append(dt*500)
    # 
    # 
    # px_all=[]
    # entropy_all=[]
    # for i,dt in zip(range(n_element),dt_list):
    #     px=compute_px(np.array(h_prot[i]),4.0)
    #     px_all.append(px)
    #     entropy_all.append(shannon_entropy(px))
    #===========================================================================
    
    
    with open('../data/tmp.pkl','rb') as f:
        [fid_all,h_prot]=pickle.load(f)
        
    pca_var=[]
    pos=14
    pca=PCA(n_components=40)
    X=np.array(h_prot[pos])
    print("dt = %.4f, T = % .2f"%(dt_list[pos],dt_list[pos]*500))
    X=pca.fit_transform(X)
    tsne=TSNE(n_components=2,perplexity=50)
    X=tsne.fit_transform(X)
    exit()
    X=np.fromfile('result.dat').reshape(-1,2)
    
    optimal_fid=np.max(fid_all[pos])
    print("Best fidelity: %.4f"%optimal_fid)
    n_sample=fid_all[0].shape[0]
    best_fid_pos=np.argsort(fid_all[pos])[int(.95*n_sample):]
    
    #Z=fid_all[pos]
    #fig, ax = plt.subplots()

    #phi_m = linspace(0, 2*pi, 100)
    #phi_p = linspace(0, 2*pi, 100)
    #X1,Y1 = np.mgrid(X[best_fid_pos,0], X[best_fid_pos,1])

    #p = ax.pcolor(X1,Y1,fid_all[pos][best_fid_pos],cmap=cm.coolwarm)#,vmin=np.min(Z),vmax=np.max(Z))
    #cb = fig.colorbar(p)
    
    #plt.show()
    plotting.density_map(X[best_fid_pos,0],X[best_fid_pos,1],fid_all[pos][best_fid_pos],title='$t=%.2f$'%(dt_list[pos]*500))
    
    #sns.
    
    #===========================================================================
    # 
    # for dt in dt_list:
    #     pca=PCA(n_components=40)
    #     X=np.array(h_prot[dt])
    #     X=pca.fit_transform(X)
    #     pca_var.append(np.sum(pca.explained_variance_ratio_[:10]))
    #     print("---> ",dt,'\t',np.sum(pca.explained_variance_ratio_[:10]))
    # 
    #===========================================================================
    
    
    #===========================================================================
    # 
    # plotting.observable([np.array(range(500)) for dt in dt_list],px_all,marker="-",labels=["%.2f"%(dt*500) for dt in dt_list],
    #                     title="Magnetization profile along the chain (n=500) for varying d$t$",xlabel='Step position',ylabel='$p(h=h_{max})$'
    #                     )
    # 
    #===========================================================================
    #===========================================================================
    # 
    # plotting.observable([dt*500 for dt in dt_list],np.array(entropy_all)/500.,marker="o-",
    #                 title="Shannon entropy of the chain (n=500) for varying d$t$",xlabel='t',ylabel='$H(p)$'
    #                 )
    # 
    #===========================================================================
    
    
    
    
    
    
    #===========================================================================
    # pca_var=[]
    # for dt in dt_list:
    #     pca=PCA(n_components=40)
    #     X=np.array(h_prot[dt])
    #     X=pca.fit_transform(X)
    #     pca_var.append(np.sum(pca.explained_variance_ratio_[:10]))
    #     print("---> ",dt,'\t',np.sum(pca.explained_variance_ratio_[:10]))
    #===========================================================================
    
    #pos=6
    #pca=PCA(n_components=50)
    #X=np.array(h_prot[pos])
    #X=pca.fit_transform(X)
    #tsne=TSNE(n_components=2, perplexity=50.0,n_iter=1000)
    #X=tsne.fit_transform(X)
    #print("Explained variance:\t",np.sum(pca.explained_variance_ratio))

    #plotting.density_map(X[:,0],X[:,1],fid_all[pos],title='$t=%.2f$'%(x[pos]))
    
    
    
    #===========================================================================
    # plotting.observable(np.array(x),np.array(pca_var),
    #                      title='Std. of fidelity for nstep$=500$ \n with variable d$t$',
    #                      xlabel='$t$',ylabel='\# of fidelity eval.')
    #===========================================================================
    
    # 
    #tsne=TSNE()
    # #X=tsne.fit
    
            
    #===========================================================================
    # plotting.observable(np.array(x),np.array(std_fid),
    #                     title='Std. of fidelity for nstep$=500$ \n with variable d$t$',
    #                     xlabel='$t$',ylabel='\# of fidelity eval.')
    #===========================================================================

    
    





    
    #===========================================================================
    # X=h_prot[4]  
    # pca=PCA(n_components=40)
    # X=pca.fit_transform(X)
    # 
    # tsne=TSNE()
    # #X=tsne.fit
    #===========================================================================
    
    
    #===========================================================================
    # 
    # pca=PCA(n_components=40)
    # pca.fi
    #===========================================================================



def shannon_entropy(px):
    return -(np.dot(px,np.log(px+0.000000000000001))+np.dot((1.-px),np.log(1.-px+0.000000000001)))

def compute_px(h_protocol,hmax):
    X=h_protocol*(0.5/hmax)+0.5 #Reshape such that all h lie between 0 and 1.
    return np.mean(X,axis=0)

if __name__=='__main__':
    main()