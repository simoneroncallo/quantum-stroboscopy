# This script contains the module to resize and plot the arrival time distributions
# and the discrete probability distribution obtained from quantum stroboscopy or the 
# non-instantaneous POVM
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Image, display

def resize_plot(fig, filepath: str, width: int, show: bool) -> None:
    """
    Save a Matplotlib figure in .png and .svg formats and 
    optionally display a resized image.
    ----------
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure
    filepath : str
        File path for saving (without extension)
    width : int
        Width for the preview image
    show : bool
        Flag to preview the saved plot, by default False
    """
    Path("./.tmp").mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath + '.png', bbox_inches='tight')
    fig.savefig(filepath + '.svg', bbox_inches='tight')
    plt.close(fig) 
    if show == True: display(Image(filename=filepath + '.png', width=width)) # Plot image of smaller size

def plot_time(user_setup, pdf: np.ndarray, probs: list | None = None, probs_labels: list | None = None, 
              small_plot = False, name: str = 'time', show : bool = False) -> None:
    """
    Plot the arrival time distribution and the discrete probability distribution (either obtained from ideal quantum 
    stroboscopy or the non-instantaneous POVM), with given detector position.
    ----------
    Parameters
    ----------
    user_setup : Stroboscopy
        Quantum stroboscopy setup (used for importing timebins)
    pdf : np.ndarray
        Discretized arrival time distribution |Ψ(T|x)|^2
    probs : list, optional
        List of arrays of discrete probabilities (indexed on the detector position)
    probs_labels : list, optional
        Plots labels for probs
    small_plot : bool, optional
        Flag to produce a square plot, by default False
    name : str, optional
        File name for saving the plot at "./.tmp/filename" (in .svg and .png), by default 'time'
    show : bool, optional
        Flag to show the saved plot, by default False
    """
    # Parameters
    size = 16
    plt.rcParams.update({'font.size': size})
    plt.rcParams["font.family"] = "serif"

    if small_plot == False: figsize = (6,5) # Standard plot
    elif small_plot == True: figsize = (5,5) # Square plot

    # Barplots
    fig, ax = plt.subplots(1,1, figsize=figsize, dpi=200, constrained_layout=True)
    # width = 1 # Fixed width
    if len(probs) == 1: 
        offsets = [0]
        width = (user_setup.discr_T[1] - user_setup.discr_T[0]) - 0.05 # Adaptive width
    elif len(probs) == 2:
        width = (user_setup.discr_T[1] - user_setup.discr_T[0])/2 - 0.05 # Adaptive width
        offsets = [-width/2, width/2]
    else: 
        plt.close(fig)
        raise ValueError('Number of barplots must be <= 2')
    for t0, prob, label in zip(offsets, probs, probs_labels): # Shape (bins_X.size,bins_T.size)
        ax.bar(user_setup.bins_T + t0, prob[:], width = width, alpha = 0.5, label = label) # Barplot

    # Distribution
    ax.plot(user_setup.fine_T, pdf[:], color = 'black', label = 'Theoretical')

    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    if small_plot == False: ax.legend(loc = 'upper right')
    
    filepath = f"./.tmp/{name}"
    resize_plot(fig, filepath, 400, show)