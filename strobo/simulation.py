# This module contains the class that performs Monte Carlo sampling for 
# non-instantaneous stroboscopic measurements of time. The sampler combines
# rejection sampling with inverse transform sampling to first draw
# the ideal measurement results and pass them though the POVM
import numpy as np
import scipy as sp
from typing import Callable
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colormaps
from .stroboscopy import Stroboscopy
from .visualize import resize_plot

class MonteCarlo():
    """
    Monte Carlo sampler for non-instantaneous stroboscopic measurements of time, 
    described by a POVM. Support limited to Gaussian components.
    ----------
    Attributes
    ----------
    rng : np.random.Generator
        Random number generator
    inf : float
        Minimum time value
    sup : float
        Maximum time value
    bins : np.ndarray
        Time bins of the stroboscoppic measurement
    stds : np.ndarray
        Standard deviations of each POVM component
    wavef : Callable
        Probability density function (pdf) of the wavepacket |Ψ(x,t)|^2
    timedistr : Callable
        Normalized arrival time distribution
    norm : float
        Normalization factor (computed by self._normalize_povm)
    povm : Callable
        Conditional probability function p(i|t) that corresponds to the POVM
    prob_failure : float
        Probability of detection failure
    """
    def __init__(self, detect: int, psisq: Callable[[np.float64, np.float64], np.float64], user_setup: Stroboscopy, \
                 width: float, generator: np.random.Generator) -> None:
        """
        Initialize and normalize the arrival time distribution and the POVM.
        ----------
        Parameters
        ----------
        detect : int
            Index of detection site
        psisq : Callable
            Probability density function (pdf) of the wavepacket |Ψ(x,t)|^2
        user_setup : Stroboscopy
            Quantum stroboscopy setup (used for importing time bins)
        width : float
            Width of each POVM component
        generator : np.random.Generator
            Random number generator
        """
        self.rng = generator
        self.inf = user_setup.min_T
        self.sup = user_setup.max_T
        self.bins = user_setup.bins_T # Time bins # Shape (M)
        self.stds = np.ones_like(user_setup.bins_T)*width # Standard deviations # Shape (M)
        self.wavef = psisq # Probability density |Ψ(x,t)|^2
        self.timedistr = self._get_pdf(user_setup.bins_X[detect], int(1e+4)) # Arrival time distribution |Ψ(t|x)|^2

        # Normalize
        self.norm = 1
        self.povm = lambda T: np.exp(-((self.bins[:,None] - T[None,:])/\
                                       self.stds[:,None])**2)/self.norm # Probability p(i|t) # Shape (M,T.size)
        self._normalize_povm(self.bins , int(1e+6))

    def _normalize_povm(self, bins: np.ndarray, num_points: int) -> None:
        """ 
        Normalize p(i|t) over time (t), by computing the maximum normalization factor
        across all time bins (i). Loose bounds lead to higher probability of detection failure.
        ----------
        Parameters
        ----------
        bins : np.ndarray
            Time bins
        num_points : int
            Number of points for numerical evaluation
        """
        times = np.linspace(self.inf, self.sup, num_points)
        norms = np.sum(self.povm(times),axis=0) # Sum over time
        self.norm = np.max(norms) # Normalize over the maximum factor
    
    def _get_pdf(self, x: np.float64, num_points: int) -> Callable[[np.float64, np.float64], np.float64]:
        """ 
        Compute the arrival time distribution predicted by the quantum clock 
        proposal [PRL 124, 110402 (2020)].
        ----------
        Parameters
        ----------
        x : float
            Detector position
        num_points : int
            Number of points for numerical integration (normalization)
        -------
        Returns
        -------
        Callable
            Normalzied arrival time distribution |Ψ(t|x)|^2
        """
        times = np.linspace(self.inf, self.sup, num_points)
        func = lambda t: self.wavef(x, t) 
        norm, _ = sp.integrate.quad(func, times[0], times[-1]) # Normalize
        return lambda t: self.wavef(x, t)/norm # Normalized time distribution
        
    def get_samples(self, user_setup: Stroboscopy, num_samples:int) -> np.ndarray:
        """
        Generate samples from the (ideal) arrival time distribution using rejection sampling. 
        Apply the (non-instantaneous) POVM with inverse transform sampling on drawn data points.
        ----------
        Parameters
        ----------
        user_setup : Stroboscopy
            Quantum stroboscopy setup (used for importing time bins)
        num_samples : int
            Number of samples
        -------
        Returns
        -------
        probs_real : np.ndarray
            Normalized probabilities of observed time bins
        """
        # Draw ideal results with rejection sampling
        samples = [] 
        while len(samples) < num_samples:
            u = self.rng.uniform(self.inf, self.sup, size=1) # Draw the independent variable
            y = self.rng.uniform(0, 1, size=1) # Draw the dependent variable
            if y < self.timedistr(u):
                samples.append(u)
        samples = np.array(samples)[:,0] # Squeeze from (num_samples,1) to (num_samples)
        
        cond_prob = self.povm(samples) # Conditional probability # Shape (M,T.size)
        cond_cumul = np.cumsum(cond_prob, axis=0) # Cumulative distribution # Shape (M,T.size)

        # Draw real results with inverse transform sampling
        U = self.rng.uniform(0, 1, size = num_samples)
        idxs = np.array([np.searchsorted(cond_cumul[:, idx_t], U[idx_t], side='left') for idx_t in range(num_samples)]) # Indexes 0 <= idx <= M
        failure_idx = self.bins.size # Identify M as a 'failure' bin
        self.prob_failure = idxs[idxs == failure_idx].size/num_samples
        success_idxs = idxs[idxs != failure_idx] # Indexes 0 <= idx <= M-1
        samples_real = self.bins[success_idxs] # Isolate 'success'

        # Compute frequencies from counts
        uniq = np.unique_counts(samples_real)
        freqs = uniq.counts/success_idxs.size # Renormalize to 'success'
        results = dict(zip(uniq.values, freqs))
        for key in self.bins:
            if key not in results:
                print(f'Adding null entry for t={key}')
                results[key] = 0 # Fix shape errors by keeping bins with null counts 
        probs_real = np.array([results[key] for key in self.bins]) # Shape (M)

        delta_T = user_setup.discr_T[1]-user_setup.discr_T[0] # Discretization step
        probs_real /= delta_T # Normalize with respect to the density
        return probs_real

    def plot_povm(self, num_points: int, small_plot: bool = False, name: str = 'povm', show: bool = False) -> None:
        """
        Plot the arrival time distribution and the POVM components.
        ----------
        Parameters
        ----------
        num_points : int
            Number of points for plotting
        small_plot : bool, optional
            Flag to produce a square plot, by default False
        name : str, optional
            File name for saving the plot at "./.tmp/filename" (in .svg and .png), by default 'povm'
        show : bool, optional
            Flag to show the saved plot, by default False
        """
        # Parameters
        size = 16
        plt.rcParams.update({'font.size': size})
        plt.rcParams["font.family"] = "serif"

        T = np.linspace(self.inf, self.sup, num_points)
        if small_plot == False: figsize = (6,5) # Standard plot
        elif small_plot == True: figsize = (5,5) # Square plot

        fig, ax = plt.subplots(1,1, figsize=figsize, dpi=200, constrained_layout=True)
        ax.plot(T, self.timedistr(T), color = 'black', linestyle = '-', label='Quantum clock')
        ax.plot(T, self.povm(T).sum(axis=0), color = 'tab:orange', linestyle = '--', label='POVM')

        ax.vlines(x = self.bins, ymin = 0, ymax = 1, linestyle = ':', color = 'tab:blue')
        
        ax.set_ylim(top=1.3)
        ax.set_xlabel("Time")
        ax.set_ylabel("Probability")
        if small_plot == False: 
            ax.legend(loc = 'upper right')
            ax.text(0.03, 0.96, f"Failure at {int(self.prob_failure*100)}%",transform=ax.transAxes, verticalalignment="top", \
                    horizontalalignment="left", fontsize=size, bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))

        filepath = f"./.tmp/{name}"
        resize_plot(fig, filepath, 400, show)