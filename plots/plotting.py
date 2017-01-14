'''
Created on Nov 7, 2016

@author: robertday

Purpose:
    Plotting functions for the different quantities of interest    
'''

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#import latex

def main():
    '''
    Example run
    ''' 
    
    # Import pkl file
    t=25
    action_set=2
    L=1
    
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
        
def protocol(protocol_array,time_slice,title=None,out_file=None,labels=None,show=True,ylabel='$h_x(t)$',xlabel="$T$"):
    """
    Purpose:
        Plots protocol vs time in latex form
    """
    
    protocols=adjust_format(protocol_array)
    
    n_curve=len(protocols)
    palette = np.array(sns.color_palette('hls',n_curve))
    fontsize=15
    
    ext_ts=np.hstack((time_slice,time_slice[-1]+time_slice[1]-time_slice[0]))
    
    if labels is not None:
        for i,p in zip(range(n_curve),protocols):
            ext_p=np.hstack((p,p[-1]))
            plt.step(ext_ts,ext_p,'-',clip_on=False,c=palette[i],label=labels[i],where='post')
            plt.plot(time_slice,p,'o',clip_on=False,c=palette[i])
        plt.legend(loc='best', shadow=True)
        
    else:
        for i,p in zip(range(n_curve),protocols):
            ext_p=np.hstack((p,p[-1]))
            plt.step(ext_ts,ext_p,'-',clip_on=False,c=palette[i],where='post')
            plt.plot(time_slice,p,'o',clip_on=False,c=palette[i])
        
    if title is not None:
        plt.title(title,fontsize=fontsize)
    

    plt.xlim([np.min(ext_ts),np.max(ext_ts)])
    if xlabel is not None:
        plt.xlabel(xlabel,fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel,fontsize=fontsize)
        
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close()
    
def observable(yarray,xarray,title=None,out_file=None,
               ylabel=None,xlabel=None,
               show=True,labels=None,
               marker="o-"
               ):
    """
    Purpose:
        Plots an observable in latex format
        
    Parameter
    ------------
    
    yarray: 1D-array -- OR  --- list OF arrays --  OR -- 2D-array
        Data component along the y-axis. Multiple data series can be passed, but this must be done
        in one of the two available formats, list of arrays of 2D-array.
        -- Warning : do not try to pass an array of arrays ! 
        
    xarray: Same as yarray but along x-axis (must match the dimensions of y !)
    
    """
    fontsize=15
    
    yarray_=adjust_format(yarray)
    xarray_=adjust_format(xarray)
   
    n_curve=len(yarray)
    palette=np.array(sns.color_palette('hls',n_curve))
    
    if labels is not None:
        for y,x,c,l in zip(yarray_,xarray_,palette,labels):
            plt.plot(x,y,marker,c=c,clip_on=False,label=l)
        plt.legend(loc='best', shadow=True,fontsize=fontsize)
    else:
        for y,x,c in zip(yarray_,xarray_,palette):
            plt.plot(x,y,marker,c=c,clip_on=False)
        
    if title is not None:
        plt.title(title,fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel,fontsize=fontsize)
    if ylabel is not None:
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