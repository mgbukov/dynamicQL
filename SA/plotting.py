'''
Created on Nov 7, 2016

@author: alexday

Purpose:
    Plotting functions for the different quantities of interest    
'''

import seaborn as sns
import numpy as np
import os
#os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.tick_params(labelsize=16)

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
    

def compare_protocols(Data,strg,save_name,save_dir):

    # change seaborn backaground colour
    sns.set(rc={'axes.facecolor':'grey', 'figure.facecolor':'grey'})

    times=Data[:,0]

    # figure size
    plt.figure(figsize=(8,4))

    

    t_max = times[-1]
    plt.xlim([0,t_max+0.01])

    fontsz=22

    plt.xlabel('$t$', fontsize=fontsz)
    if strg=='protocols':


        plt.step(times,Data[:,1],color='orange',marker='.',where='post',linewidth=1,label='bang-bang')
        plt.step(times,Data[:,5],color='yellow',marker='.',where='post',linewidth=1,label='quasi-cont')
        plt.step(times,Data[:,9],'r',marker='.',where='post',linewidth=1,label='LZ')
        plt.step(times,Data[:,13],'b',marker='.',where='post',linewidth=1,label='geodesic')

        plt.ylabel('$h(t)$', fontsize=fontsz)
        
        p_max = max(Data[:,1])
        p_min = min(Data[:,1])
        plt.ylim([p_min-0.5,p_max+0.5])
        # set fig aspect ratio
        #plt.axes().set_aspect(0.05)


    else:

        plt.step(times,Data[:,4],color='orange',marker='.',where='post',linewidth=1,label='bang-bang')
        plt.step(times,Data[:,8],color='yellow',marker='.',where='post',linewidth=1,label='quasi-cont')
        plt.step(times,Data[:,12],'r',marker='.',where='post',linewidth=1,label='LZ')
        plt.step(times,Data[:,16],'b',marker='.',where='post',linewidth=1,label='geodesic')


        plt.ylabel('$F_h(t)=|\langle\psi_\\ast|\psi(t)\\rangle|^2$', fontsize=fontsz)
    
    if strg=='fidelities':
        plt.legend(loc='upper left', fontsize=fontsz-6)

    plt.tick_params(labelsize=fontsz-2)
    plt.grid(True)

    # avoids x axis label being cut off
    plt.tight_layout()


    save_str = strg+save_name+'.pdf'
    plt.savefig(save_dir+save_str)

    #show()
    plt.close()    


def protocol(time_slice,protocol_array,title=None,out_file=None,labels=None,show=True,ylabel='$h_x(t)$',xlabel="$t$"):
    """
    Purpose:
        Plots protocol vs time in latex form
    """
    
    protocols=adjust_format(protocol_array)

    # fig size
    plt.figure(figsize=(8,4))
    
    n_curve=len(protocols)
    palette = np.array(sns.color_palette('hls',n_curve))
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

def protocol_ising_2D_map(protocol_list, show=True):
    sns.set_style("whitegrid", {'axes.grid' : False})
    hx=np.vstack(protocol_list)
    # Z is your data set
    N = len(hx)
    G = np.zeros((N,hx.shape[1],3))

    # Where we set the RGB for each pixel
    G[hx > 0] = [1,1,1]
    G[hx < 0] = [0,0,0]
    fig=plt.imshow(G,interpolation='nearest',aspect='auto')
    if show is True:
        plt.show()


def observable(xarray,yarray,title=None,out_file=None,
               xlim=None,ylim=None,
               ylabel=None,xlabel=None,linewidth=1,
               show=True,labels=None,legend_loc='best',
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
    fontsize=26
    
    yarray_=adjust_format(yarray)
    xarray_=adjust_format(xarray)
   
    n_curve=len(yarray)
    palette=np.array(sns.color_palette('hls',n_curve))
    
    if labels is not None:
        for y,x,c,l in zip(yarray_,xarray_,palette,labels):
            plt.plot(x,y,marker,c=c,clip_on=False,label=l,linewidth=linewidth)
        plt.legend(loc=legend_loc, shadow=True,fontsize=fontsize-2)
    else:
        for y,x,c in zip(yarray_,xarray_,palette):
            plt.plot(x,y,marker,c=c,clip_on=False,linewidth=linewidth)
        
    if title is not None:
        plt.title(title,fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel,fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel,fontsize=fontsize)
        
    plt.tick_params(labelsize=22)
    
    if xlim: plt.xlim(xlim);
    if ylim: plt.ylim(ylim);

    # avoids x axis label being cut off
    plt.tight_layout()

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