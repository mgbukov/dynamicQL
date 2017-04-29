# dynamicQL

This branch contains a Python implementation of Reinforcement Learning for performing quantum state preparation on a 1D chain.

# Requirements

Non-standard packages:
- [Quspin](https://github.com/weinbe58/QuSpin#installation)

# To run #

`main_RL.py`: defines the physics and RL parameters.

`Q_learning.py`: implemeents a modified version of Watkin's Q-Learning [this code can be factored but performance will slow down; future versions will try to cython-ise parts of it].

`Hamiltonian.py`: builds the Hamiltonian and pre-calculates the matrix exponentials for all protocol steps.  

