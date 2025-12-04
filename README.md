# Quantum stroboscopy
A Python package to perform Monte Carlo simulations of <i>quantum stroboscopy</i>, i.e. strong position measurements on multiple copies of the system at different times. It supports both ideal and non-instantaneous setups. The latter is implemented through a sequence of Gaussian functions, each one representing a different time bin of the stroboscopic measurement. The simulation combines
rejection sampling with inverse transform sampling to first draw the ideal measurement results and pass them through the positive operator-valued measure (POVM).

[![arXiv](https://img.shields.io/badge/arXiv-2507.17740-b31b1b.svg)](https://doi.org/10.48550/arXiv.2507.17740)

Contributors: Simone Roncallo [@simoneroncallo](https://github.com/simoneroncallo) <br>
Reference: Seth Lloyd, Lorenzo Maccone, Lionel Martellini, Simone Roncallo <i>“Quantum stroboscopy for time measurements”</i> [arXiv:2507.17740](https://doi.org/10.48550/arXiv.2507.17740)

## Installation
The package can be downloaded and set up in a [conda](https://docs.conda.io/) environment with
```bash
conda create --name arrival-env
conda activate quantum-stroboscopy
conda install --file requirements.txt
```
or containerized in [Docker](https://docs.docker.com/) with
```bash
sudo docker build -t jupyter-time .
./runDocker.sh
```

## Example
Initialize a Gaussian wave packet with 
```python
from strobo import get_gaussian
x0 = -4 # Position (t=0)
p = 3 # Momentum
std = 1 # Standard deviation (t=0)
mass = 1 # Mass
wave_packet = get_gaussian(x0, p, std, mass)
```
Simulate the non-instantaneous stroboscopic measurement as
```python
from strobo import Stroboscopy, MonteCarlo
rng = np.random.default_rng(2025)
size = 10000 # Sample size
width = 0.35 # Width of each POVM component
strobo = Stroboscopy(numT, numX, intT[0], intT[1], intX[0], intX[1])
povm = MonteCarlo(idx, wave_packet, strobo, width, rng)
results = povm.get_samples(strobo, size)
```
with `numT`, `numX` the number bins in the intervals `intT` and `intX`, and with `idx` labelling the detector bin.<br><br>
<img src="https://github.com/simoneroncallo/quantum-stroboscopy/blob/main/images/example.svg/">

## Structure
This repository has the following structure
```bash
strobo/
    ├── simulation.py # Monte Carlo simulation
    ├── stroboscopy.py # Stroboscopic setup
    └── visualize.py

example.ipynb # Example notebook
requirements.txt # Dependencies
runDocker.sh # Container
```

