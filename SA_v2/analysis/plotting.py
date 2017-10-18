import numpy as np
from matplotlib import pyplot as plt
import sys,os
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing as prep

    
def density_map(X, kde, savefile='test.png', show=True, xlabel=None, ylabel=None, n_mesh = 400, vmin = None, vmax= None, compute_zmax = False):

    plt.rc('text', usetex=True)
    font = {'family' : 'serif', 'size': 18}
    plt.rc('font', **font)

    fig =  plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111) 
    #n_mesh=400
    xmin, xmax = np.percentile(X[:,0],q=10.),np.max(X[:,0])
    #xmin, xmax = np.min(X[:,0]),np.max(X[:,0])
    dx = xmax - xmin
    ymin, ymax = np.min(X[:,1]),np.max(X[:,1])
    dy = ymax - ymin

    print('density map')
    x = np.linspace(xmin-0.1*dx,xmax+0.1*dx, n_mesh)
    y = np.linspace(ymin-0.1*dy,ymax+0.1*dy, n_mesh)
    extent = (xmin-0.1*dx, xmax+0.1*dx, ymin-0.1*dy, ymax+0.1*dy)

    mms=prep.MinMaxScaler()
<<<<<<< HEAD
    my_map=plt.get_cmap(name='BuGn')
=======
>>>>>>> 12f652fab3c5e344474bb431ecc0a7e3a8454d01

    xy=np.array([[xi, yi] for yi in y for xi in x])
    #print("kk")
    z = np.exp(kde.evaluate_density(xy)) # this should be the computationally expensive part 
    #z = np.exp(rho)
    #print("ksdjfk")
    zmax = np.max(z)
    if compute_zmax is True:
        return zmax
    
    if vmax is None:
        vmax = zmax

<<<<<<< HEAD
    print('density map')
    Zrgb[Z < 0.005] = (1.0,1.0,1.0,1.0)

    plt.imshow(Zrgb, interpolation='bilinear',cmap='BuGn', extent=extent,origin='lower', aspect='auto', zorder=1)
    cb=plt.colorbar()
    cb.set_label(label='Density',labelpad=10)
    print("-----")   
 
    X1, Y1 = np.meshgrid(x,y)
    plt.contour(X1, Y1, Z, levels=np.linspace(0.03,0.8,6), linewidths=0.3, colors='k', extent=extent, zorder=2)
=======
    #z=mms.fit_transform(z.reshape(-1,1))
    Z=z.reshape(n_mesh, n_mesh)
    Z[Z < 0.005] = 0
    

    plt.imshow(Z, interpolation='bilinear',cmap='BuGn',  extent=extent, aspect='auto', origin='lower', zorder=1, vmin=vmin,vmax=vmax)
    cb = plt.colorbar()
    cb.set_label(label='Density',labelpad=10)
    
    X1, Y1 = np.meshgrid(x,y)
    plt.contour(X1, Y1, Z, levels=np.linspace(0.03*(vmax-vmin),vmax,6), linewidths=0.3, colors='k', extent=extent, zorder=2)

    #plt.contour(X1, Y1, Z, levels=[0.001*(vmax-vmin)], linewidths=0.3, colors='blue',ls='dashed', extent=extent, zorder=2)

    ax.grid(False)

>>>>>>> 12f652fab3c5e344474bb431ecc0a7e3a8454d01
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
    plt.rc('text', usetex=True)
    font = {'family' : 'serif', 'size'   : 16}
    plt.rc('font', **font)

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
            plt.step(ext_ts,ext_p,'-',clip_on=False,c='black',where='post',zorder=0,lw=5)
            #plt.scatter(time_slice,p,clip_on=False,c='black',marker='o',s=8,edgecolor='black',linewidths=0.5,zorder=1)
            #plt.plot(time_slice,p,clip_on=False,c=palette[i])
        
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

def smooth_MA(y, window=20, padding = 'left'):
    if padding == 'left':
        yi = y[0]
        yf = y[-1]
        ypad = np.concatenate((yi*np.ones(window,dtype=float),y))
        ysmooth = np.zeros(len(y),dtype=float)

        for i in range(len(y)):
            ysmooth[i]=np.mean(ypad[i:window+i])
    
    return ysmooth

def smooth_data(x,y): # smooth data using moving average --> 
    from scipy.optimize import curve_fit
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter

    xx = np.linspace(x.min(),x.max(), 1000)

    # interpolate + smooth
    itp = interp1d(x,y, kind='linear')
    window_size, poly_order = 101, 4
    yy_sg = savgol_filter(itp(xx), window_size, poly_order)

    return xx, yy_sg


def density_trajectory(final_fid, c=(0.229527,0.518693,0.726954), show=True):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    from sklearn.neighbors.kde import KernelDensity

    kde = KernelDensity(kernel='gaussian', bandwidth=0.006).fit(final_fid[:, np.newaxis])

    fig, ax = plt.subplots(1, 1, sharex=True)

    x=np.linspace(0,1.1,5000)
    y=np.exp(kde.score_samples(x[:, np.newaxis]))
    ax.fill_between(x, 0, y, facecolor=c)
    ax.plot(x,y,c=c)
    #plt.yscale('log')
    plt.xlim((-0.05,1.05))
    #print(show)
    if show is True:
        plt.show()

def density_trajectory_2(final_fid, c_list, idx_list, show=True):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    from sklearn.neighbors.kde import KernelDensity

    
    fig, ax = plt.subplots(1, 1, sharex=True)
    X=final_fid[:,np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.006).fit(X)
    x=np.linspace(0,1.1,5000)
    y=np.exp(kde.score_samples(x[:, np.newaxis]))

    i1 = [0,1000]
    i2 = [1000,4000]
    i3 = [4000,5000]

    c0 = c_list[0]
    ax.fill_between(x[i1[0]:i1[1]], 0, y[i1[0]:i1[1]], facecolor=c0)
    ax.plot(x[i1[0]:i1[1]],y[i1[0]:i1[1]],c=c0)
    c1 = c_list[1]
    ax.fill_between(x[i2[0]:i2[1]], 0, y[i2[0]:i2[1]], facecolor=c1)
    ax.plot(x[i2[0]:i2[1]],y[i2[0]:i2[1]],c=c1)
    c2 = c_list[2]
    ax.fill_between(x[i3[0]:i3[1]], 0, y[i3[0]:i3[1]], facecolor=c2)
    ax.plot(x[i3[0]:i3[1]],y[i3[0]:i3[1]],c=c2)

    #ax.plot(x,y,c=c_list[i])
    #plt.yscale('log')
    plt.xlim((-0.05,1.05))
    #print(show)
    if show is True:
        plt.show()


def trajectory(traj,c_idx):

    plt.rc('text', usetex=True)
    font = {'family' : 'serif', 'size'   : 16}
    plt.rc('font', **font)
    
    green= (0.477789,0.719150,0.193583)
    blue = (0.229527,0.518693,0.726954)
    red = (0.797623,0.046473,0.127759)
    clist = [green,blue,red]
    for i,s in enumerate(traj):
        #s_smooth=smooth_MA(s)    
        plt.plot(range(len(s)),s,c=clist[c_idx[i]],zorder=2-c_idx[i])
    for i,s in enumerate(traj):
        plt.scatter(len(s)-1,s[-1],zorder=3,c=clist[c_idx[i]],marker='o',s=15,edgecolor='black',linewidths=0.5)
        #plt.scatter(len(s)-1,s[-1],zorder=3,c='black',marker='o',s=15,edgecolor='black',linewidths=0.5)

    plt.xlabel('SD iterations')
    plt.ylabel('Fidelity')
    #plt.ylim((-0.05,1.1))
    plt.xlim((-10,1200))
    plt.show()

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
