'''
Created on Nov 7, 2016

@author: robertday

Purpose:
    Plotting functions for the different quantities of interest    
'''

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import latex

def main():
    '''
    Example run
    ''' 
    
    # Import pkl file
    t=25
    action_set=2
    L=1
    
    #===========================================================================
    # fid_vs_t=[]
    # for tt in [5,10,15,20,25,30,35,40,45,50]:
    #     file=open('data/a%d_t%d.pkl'%(action_set,tt),'rb')
    #     data=pickle.load(file,encoding='latin1')
    #     print(np.mean(np.array([d[0] for d in data])))
    #     fid_vs_t.append([tt*0.05,np.mean(np.array([d[0] for d in data]))])
    # 
    # title="Fidelity vs quench time for $L=1$ \n $a\in\\{0,2\\}$, $h_z=1.0$"#\pm0.02,\pm0.02,\pm 0.1,\pm 0.2,\pm 0.5,\pm 1.0,\pm 2.0\\}$, $h_z=1.0$"
    # out_file="plots/FidvsT_L=1_a=1.pdf"
    # plot_fidelity(np.array(fid_vs_t),title=title,out_file=out_file)
    #===========================================================================



def protocol(protocol_array,time_slice,title=None,out_file=None,labels=None,show=False,ylabel='$h_x(t)$',xlabel="$t$"):
    """
    Purpose:
        Plots protocol vs time in latex form
    """
    
    n_curve=protocol_array.shape[0]
    palette = np.array(sns.color_palette('hls',n_curve))
    fontsize=15
    
    ext_ts=np.hstack((time_slice,time_slice[-1]+time_slice[1]-time_slice[0]))
    
    if labels is not None:
        for i,p in zip(range(n_curve),protocol_array):
            ext_p=np.hstack((p,p[-1]))
            plt.step(ext_ts,ext_p,'-',clip_on=False,c=palette[i],label=labels[i],where='post')
            plt.plot(time_slice,p,'o',clip_on=False,c=palette[i])
        plt.legend(loc='best', shadow=True)
        
    else:
        for i,p in zip(range(n_curve),protocol_array):
            ext_p=np.hstack((p,p[-1]))
            plt.step(ext_ts,ext_p,'-',clip_on=False,c=palette[i],where='post')
            plt.plot(time_slice,p,'o',clip_on=False,c=palette[i])
        
    if title is not None:
        plt.title(title,fontsize=fontsize)
    

    plt.xlim([np.min(ext_ts),np.max(ext_ts)])
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
        
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close()
    
def observable(yarray,xarray,title=None,out_file=None,ylabel="$F$",xlabel="$t$",show=False,labels=None):
    """
    Purpose:
        Plots an observable in latex format
        
    Parameter
    ------------
    
    yarray: 1D-array or list of arrays or 2D-array
        Data component along the y-axis. Multiple data series can be passed, but this must be done
        in one of the two available formats, list of arrays of 2D-array.
        -- Warning : do not try to pass an array of arrays ! 
        
    xarray: Same as yarray but along x-axis (must match the dimensions of y !)
    
    """
    fontsize=15
    
    if isinstance(yarray,list) is False:    
        if len(yarray.shape)==1:
            yarray=yarray.reshape(1,-1)
        assert len(yarray.shape)==2, "Y has the wrong shape"
        n_curve=yarray.shape[0]
    else:
        n_curve=len(yarray)
    
    if isinstance(xarray,list) is False:
        if len(xarray.shape)==1:
            xarray=xarray.reshape(1,-1)
        assert len(xarray.shape)==2, "X has the wrong shape"
        
    palette =np.array(sns.color_palette('hls',n_curve))
    
    if labels is not None:
        for y,x,c,l in zip(yarray,xarray,palette,labels):
            plt.plot(x,y,"-",c=c,clip_on=False,label=l)
        plt.legend(loc='best', shadow=True)
    else:
        for y,x,c in zip(yarray,xarray,palette):
            plt.plot(x,y,"-",c=c,clip_on=False)
        
    if title is not None:
        plt.title(title,fontsize=fontsize)
    
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
        
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
        
    plt.close()
    
def visne_2D(x,y,marker_intensity,zlabel="Fidelity",out_file=None,title=None,show=False,label=None,xlabel='Dim 1',ylabel='Dim 2'):
    print("Starting viSNE plot 2D")
    fontsize=15

    """ 
    Purpose:
        2-D scatter plot of data embedded in t-SNE space (2 dimensional) with intensity levels
    
    """
    
    z=marker_intensity
    plt.scatter(x,y,c=z,cmap="BuGn",alpha=1.0,label=label)
    plt.tick_params(labelbottom='off',labelleft='off')
    cb=plt.colorbar()
    
    if label is not None:
        cb.set_label(label=zlabel,labelpad=10)
    
    if title is not None:
        plt.title(title,fontsize=fontsize)
    
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.legend(loc='upper right', shadow=True)
    
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close()
    
    
# Run main program !
if __name__ == "__main__":
    main()