import numpy as np
from matplotlib import pyplot as plt
import sys,os
sys.path.append("..")
import seaborn as sns

def contour(x,y, savefile='test.png',show=True):
    
    g = sns.jointplot(x,y, kind="kde", space=0, color="g")
    g.ax_marg_x.set_axis_off()
    g.ax_marg_y.set_axis_off()
    
    if show :
        plt.show()
    plt.savefig(savefile)