'''
Created on Nov 7, 2016

@author: robertday

Purpose:
    Plotting functions for the different quantities of interest    
'''

import seaborn as sns
sns.set_style("whitegrid")
import numpy as np
import matplotlib.pyplot as plt
import latex

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
    
def observable(xarray,yarray,title=None,out_file=None,
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
        plt.legend(loc='upper left', bbox_to_anchor=(1,1),shadow=True,fontsize=fontsize)
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
    
def density_map(x,y,z,
                xlabel=None,ylabel=None,zlabel=None,label=None,
                centers=None,
                out_file=None,title=None,show=True,cmap='coolwarm',remove_tick=False):
    """ 
    Purpose:
        Produces a 2D intensity map
    """
    palette=np.array(sns.color_palette('hls', 10))
    
    fontsize=15

    if label is not None:
        plt.scatter(x,y,c=z,cmap=cmap,alpha=1.0,rasterized=True,label=label)
    else:
        plt.scatter(x,y,c=z,cmap=cmap,alpha=1.0,rasterized=True)
    
    cb=plt.colorbar()
    
    if remove_tick:
        plt.tick_params(labelbottom='off',labelleft='off')
    
    if xlabel is not None:
        plt.xlabel(xlabel,fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel,fontsize=fontsize)
    if zlabel is not None:
        cb.set_label(label=zlabel,labelpad=10)
    if title is not None:
        plt.title(title,fontsize=fontsize)
    if label is not None:
        plt.legend(loc='best')
        
    if centers is not None:
        plt.scatter(centers[:,0],centers[:,1],s=200,marker='*',c=palette[3])
    
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
  
def DOS(list_of_values,label=None,xlabel='$F$',ylabel='$\\rho(F)$',outfile=None,show=True):
    c=sns.color_palette('hls',8)
    sns.set(font_scale=1.5)
    ax=sns.distplot(list_of_values,
                 kde_kws={"color": c[0], "lw": 1,"label":label},
                 hist_kws={"alpha":0.9,"range": [0,1]},color=c[4],
                 )
    ax.set_xlabel(xlabel,fontsize=16)
    ax.set_ylabel(ylabel,fontsize=16)
    if outfile is not None:
        plt.savefig(outfile)
    if show:
        plt.show()  
    
# Run main program !
if __name__ == "__main__":
    main()