import numpy as np
from matplotlib import pyplot as plt
import sys,os
from sklearn.neighbors import KernelDensity

def density_map(X,kde,savefile='test.png',show=True,xlabel=None,ylabel=None):

    plt.rc('text', usetex=True)
    font = {'family' : 'serif', 'size'   : 40}
    plt.rc('font', **font)

    fig =  plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111) 
    n_mesh=400
    xmin,xmax = np.percentile(X[:,0],q=10.),np.max(X[:,0])
    #xmin, xmax = np.min(X[:,0]),np.max(X[:,0])
    dx = xmax - xmin
    ymin, ymax = np.min(X[:,1]),np.max(X[:,1])
    dy = ymax - ymin

    x = np.linspace(xmin-0.1*dx,xmax+0.1*dx, n_mesh)
    y = np.linspace(ymin-0.1*dy,ymax+0.1*dy, n_mesh)
    extent = (xmin-0.1*dx,xmax+0.1*dx,ymin-0.1*dy,ymax+0.1*dy)

    from sklearn import preprocessing as prep
    mms=prep.MinMaxScaler()

    my_map=plt.get_cmap(name='BuGn')

    xy=np.array([[xi,yi] for yi in y for xi in x])
    #print("kk")
    z = np.exp(kde.evaluate_density(xy))
    #z = np.exp(rho)
    #print("ksdjfk")
    z=mms.fit_transform(z.reshape(-1,1))
    Z=z.reshape(n_mesh, n_mesh)
    z=my_map(z)
    Zrgb = z.reshape(n_mesh, n_mesh, 4)

    
    Zrgb[Z < 0.005] = (1.0,1.0,1.0,1.0)

    plt.imshow(Zrgb, interpolation='bilinear',cmap='BuGn', extent=extent,origin='lower',aspect='auto',zorder=1)
    cb=plt.colorbar()
    cb.set_label(label='Density',labelpad=10)

    X1, Y1 = np.meshgrid(x,y)
    plt.contour(X1, Y1, Z, levels=np.linspace(0.05,0.8,5), linewidths=0.3, colors='k', extent=extent,zorder=2)
    ax.grid(False)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    #plt.show()


def protocol(time_slice,protocol_array,title=None,out_file=None,labels=None,show=True,ylabel='$h_x(t)$',xlabel="$t$"):
    """
    Purpose:
        Plots protocol vs time in latex form
    """
    palette=[plt.get_cmap('Dark2')(0),plt.get_cmap('Dark2')(10),plt.get_cmap('Dark2')(20)]
    protocols=adjust_format(protocol_array)

    # fig size
    plt.figure(figsize=(8,4))
    
    n_curve=len(protocols)

    #palette = np.array(sns.color_palette('hls',n_curve))
    fontsize=15
    ext_ts=np.hstack((time_slice,time_slice[-1]+time_slice[1]-time_slice[0]))
    
    if labels is not None:
        for i,p in zip(range(n_curve),protocols):
            ext_p=np.hstack((p,p[-1]))
            plt.step(ext_ts,ext_p,'-',clip_on=False,c=palette[i],label=labels[i],where='post')
            plt.plot(time_slice,p,'o',clip_on=False,c=palette[i])
        plt.legend(loc='best', shadow=True,fontsize=fontsize)
        
    else:
        for i,p in zip(range(n_curve),protocols):
            ext_p=np.hstack((p,p[-1]))
            plt.step(ext_ts,ext_p,'-',clip_on=False,c=palette[i],where='post')
            plt.plot(time_slice,p,'o',clip_on=False,c=palette[i])
        
    if title is not None:
        plt.title(title,fontsize=fontsize)

    plt.tick_params(labelsize=fontsize)
    

    plt.xlim([np.min(ext_ts),np.max(ext_ts)])
    if xlabel is not None:
        plt.xlabel(xlabel,fontsize=fontsize+4)
    if ylabel is not None:
        plt.ylabel(ylabel,fontsize=fontsize+4)
        
    # avoids x axis label being cut off
    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close()


def adjust_format(my_array):
    if isinstance(my_array, np.ndarray):
        if len(my_array.shape)==1:
            return [my_array]
        else:
            return my_array
    elif isinstance(my_array,list):
        e1=my_array[0]
        if isinstance(e1,np.ndarray):
            return my_array
        elif isinstance(e1,list):
            return my_array
        else:
            return [np.array(my_array)]
    else:
        assert False







def main():
    print("heelo")
    #n_cluster=3
    #from sklearn.datasets import make_blobs
    #X,y=make_blobs(n_samples=1000,random_state=0,centers=n_cluster)

    #for i in range(n_cluster):
    #    Xi=X[y==i]
    #    plt.scatter(Xi[:,0],Xi[:,1])
    #plt.show()
    #kde=KernelDensity(bandwidth=0.2, algorithm='kd_tree')#, atol=0.00001, rtol=0.000001)
    #kde.fit(X)
    #z=kde.score_samples(X)
    #plt.scatter(X[:,0],X[:,1],c=z,cmap='coolwarm')
    #plt.show()
    #density_map(X)

if __name__== "__main__":
    main()