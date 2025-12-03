# This module performs the numerical computation (by integration) of the quantum
# stroboscopic measurement of time, obtained as a probability matrix
# that corresponds to the outcome of projective position measurements
# performed at different times on a Gaussian wavepacket
import numpy as np
import scipy as sp
from typing import Callable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
from .visualize import resize_plot

def get_gaussian(x0: float, p0: float, sigma: float, mass: float) -> Callable[[np.float64, np.float64], np.float64]:
    """ Compute the probability distribution of a Gaussian packet (free particle) that solves 
    the Schrodinger's equation with given initial conditions. 
    ----------
    Parameters
    ----------
    x0 : float
        Initial position
    p0 : float
        Initial momentum
    sigma : float
        Initial width (standard deviation)
    mass : float
        Mass
    -------
    Returns
    -------
    Callable[[np.float64, np.float64], np.float64]
        Packet probability density at position x and time t
    """
    def gaussian(x,t):
        # Temporary function
        arg = -(x-x0-p0*t/mass)**2/(sigma**2 + t**2/(sigma*mass)**2)
        return np.exp(arg)/np.sqrt(np.pi*(sigma**2 + t**2/(sigma*mass)**2))
    return gaussian

class Stroboscopy():
    """
    Numerical computation for quantum stroboscopic measurements of time, implemented
    as projective position measurements performed at different times on a Gaussian wavepacket.
    ----------
    Attributes
    ----------
    num_T : int
        Number of time bins
    num_X : int
        Number of position bins
    min_T : float
        Minimum time value
    max_T : float
        Maximum time value
    min_X : float
        Minimum position value
    max_X : float
        Maximum position value
    discr_T : np.ndarray
        Time grid for numerical evaluation and integration
    fine_T : np.ndarray
        Fine grid for the arrival time distribution
    discr_X : np.ndarray
        Position grid for numerical evaluation and integration
    bins_T : np.ndarray
        Time bins midpoints
    bins_X : np.ndarray
        Position bins midpoints
    prob_strobo : np.ndarray
        Discrete probability matrix for quantum stroboscopy (t,x)
    prob_time : np.ndarray
        Discretized arrival time distribution |Ψ(t|x)|^2, with detector placed at x
    """
    def __init__(self, num_T: int, num_X: int, min_T: float, max_T: float, min_X: float, max_X: float) -> None:
        self.num_T = num_T # Number of time bins
        self.num_X = num_X # Number of position bins
        self.min_T = min_T # Minimum time value
        self.max_T = max_T # Maximum time value
        self.min_X = min_X # Minimum position coordinate
        self.max_X = max_X # Maximum position coordinate

        # Generate equally-spaced points
        self.discr_T = np.linspace(0, max_T, num_T+1) # Time grid
        self.fine_T = np.linspace(0, max_T, int(num_T*1e3)) # Finer time grid
        self.discr_X = np.linspace(0, max_X, num_X+1) # Spatial grid

        # Compute bins
        self.bins_T = np.array([(self.discr_T[idx+1] + self.discr_T[idx])/2 for idx in range(num_T)])
        self.bins_X = np.array([(self.discr_X[idx+1] + self.discr_X[idx])/2 for idx in range(num_X)])

    def get_results(self, user_packet: Callable[[np.float64, np.float64], np.float64]) -> Tuple[[np.ndarray, np.ndarray]]:
        """
        Compute the results of quantum stroboscopy and the arrival time probability, by numerical integration of 
        the Gaussian wave packet. Return results as probability matrices of shape (self.num_X, self.num_T), 
        ----------
        Parameters
        ----------
        user_packet : Callable[[np.float64, np.float64], np.float64]
            Packet probability density at position x and time t
        -------
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (prob_strobo, prob_time)
        """
        res = np.zeros((self.num_X, self.num_T))
        for idx_T in range(self.num_T):
            func = lambda x: user_packet(x,self.bins_T[idx_T]) # Choose T
            for idx_X in range(self.num_X):
                res[idx_X,idx_T], _ = sp.integrate.quad(func, self.discr_X[idx_X], self.discr_X[idx_X+1]) # Integrate over X
        self.prob_strobo = res/(np.sum(res, axis=1)[:,np.newaxis]*(self.discr_T[1]-self.discr_T[0])) # Normalize
    
        self.prob_time = np.zeros((self.num_X, self.fine_T.size))
        for idx, x in enumerate(self.bins_X):
            func = lambda t: user_packet(x, t) 
            norm, _ = sp.integrate.quad(func, self.discr_T[0], self.discr_T[-1]) # Normalize
            self.prob_time[idx,:] = user_packet(x, self.fine_T)/norm # Conditional probability
        
        return self.prob_strobo, self.prob_time
    
    def plot_grid(self, name: str = 'grid', show: bool = False) -> None:
        """
        Plot the normalized stroboscopic probability matrix as 
        a (self.num_X, self.num_T) grid.
        ----------
        Parameters
        ----------
        name : str
            File name for saving the plot at "./.tmp/filename" (in .svg and .png), by default 'grid'
        show : bool
            Flag to show the saved plot, by default False
        """
        # Parameters
        size = 16
        plt.rcParams.update({'font.size': size})
        plt.rcParams["font.family"] = "serif"
    
        blues = colormaps['Blues']
        trimmed_blues = mcolors.LinearSegmentedColormap.from_list('trimmed_blues', blues(np.linspace(0,.75))) # Colormap
    
        min_vals = self.prob_strobo.min(axis=1, keepdims=True)
        max_vals = self.prob_strobo.max(axis=1, keepdims=True)
        matrix = (self.prob_strobo - min_vals) / (max_vals - min_vals) # Stroboscopy probability matrix
    
        fig, ax = plt.subplots(1, 1, figsize=(5,5), dpi=500, constrained_layout=True)
        ax.imshow(matrix, cmap=trimmed_blues, interpolation='none', vmin = 0) # Matrix
        
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlabel("Time"), ax.set_ylabel("Position")
        ax.set_xticks(np.arange(-0.5, self.prob_strobo.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.prob_strobo.shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1) # Highlight grid
        ax.tick_params(which='minor', bottom=False, left=False)
    
        filepath = f"./.tmp/{name}"
        resize_plot(fig, filepath, 400, show)