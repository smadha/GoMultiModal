import numpy as np 
from matplotlib import pyplot as plt


def plot_histogram(x, var_name):
    # histogram of x data, 10 bins equal size
    _, hist_data, _ = plt.hist(x, bins=20)
    plt.plot(x=hist_data)
    plt.savefig(var_name+"-hist", linewidth=0)
#     plt.show()
    plt.close()

def plot_xy_histogram(a, b, var_name, label):
    bins=np.histogram(np.hstack((a,b)), bins=40)[1] #get the bin edges
    plt.hist(a, bins)
    
    plt.savefig(label[0] + var_name + "-hist", linewidth=0)
    plt.close()
    
    plt.hist(b, bins)
    plt.savefig(label[1] + var_name + "-hist", linewidth=0)
    plt.close()

    


def plot_box(data_to_plot, var_name, label):
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)
    # Create the boxplot
    bp = ax.boxplot(data_to_plot)
    ## Custom x-axis labels
    ax.set_xticklabels(label)
    # Save the figure
    fig.savefig(var_name+"-box", bbox_inches='tight')
    plt.close()
    
def plot_scatter(x, var_name,axis=None):
    if axis==None:
        axis = [0, 100000, min(x)-10, max(x)+10]
    plt.axis(axis)
    plt.plot(range(0,len(x)),x,"ro",markersize=1)
    plt.savefig(var_name)
#     plt.show() 
    plt.close()
