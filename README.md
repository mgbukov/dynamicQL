# dynamic_QL

This branch contains a Python3 implementation of simulated annealing for performing quantum state preparation on a 1D chain. The code is fairly well commented (so dig in !).

# Requirements

Non-standard packages:
- [Quspin](https://github.com/weinbe58/QuSpin)
- Seaborn (install using pip or conda)

# To run #

Clone or download this repository, then run the following command :
```
python LZ_sim_anneal.py 0 80 'bang-bang8' auto 3000 0.05 200 True
```
For help use:
```
python LZ_sim_anneal.py -h
```
The meaning of the given parameters is explained in the code LZ_sim_anneal.py
