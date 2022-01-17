#%%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_confusion_matrix(true: np.ndarray, pred: np.ndarray):
    '''
    get confusion matrix for one model
    input: true labels and predicted labels - one dimensional np arrays
    output: confusion matrix'''
    M = 16
    confusion_matrix: np.ndarray = np.zeros((M,M))
    for i,j in zip(true-1,pred-1): confusion_matrix[i,j] += 1
    return confusion_matrix

def get_confusion_matrices(true, *preds):
    '''
    get multiple conf matrices - stored in a list
    takes the same inputs as the McNemar test function
    '''
    conf_matrices = [get_confusion_matrix(true,pred) for pred in preds]
    return conf_matrices

def plot_conf_matrices(conf_matrices, model_names, title = "Confusion matrices"):
    '''
    plot confusion matrices as heatmap in one plot

    input1: confusion matrices stored in a list - same format as the output from get_confusion_matrices
    input2: model names in a list of strings

    output: pyplot figure and subplot axes - should anyone wanna do more with it
    '''
    n_conf_matrices = len(conf_matrices)
    vmin = np.min(conf_matrices)
    vmax = np.max(conf_matrices)
    fig , axs = plt.subplots(1,n_conf_matrices, figsize = (3*n_conf_matrices, 3.6), constrained_layout = True)
    for i in range(n_conf_matrices):
        axs[i].set_title(model_names[i])
        plot = axs[i].imshow(conf_matrices[i], vmin = vmin, vmax = vmax)
        axs[i].set_xticks(np.arange(8)*2)
        axs[i].set_xticklabels(np.arange(8)*2+1)
        if i == 0:
            axs[i].set_yticks(np.arange(8)*2)
            axs[i].set_yticklabels(np.arange(8)*2+1)
            axs[i].set_ylabel("True experiment no.")
        else:
            axs[i].set_yticks([0])
            axs[i].set_yticklabels([""])
        if i == int(n_conf_matrices/2):
            #sorry - this was the best way to get the x-label centered haha
            axs[i].set_xlabel("Predicted experiment no.")
    

    fig.colorbar(plot, shrink = 0.78)

    fig.suptitle(title, fontsize = 17)
    plt.show()

    return fig, axs


