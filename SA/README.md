# dynamicQL

This branch contains a Python3 implementation of simulated annealing and stochastic gradient descent for performing quantum state preparation on a 1D chain. The code is fairly well commented (so dig in !).

# Requirements

Non-standard packages:
- [Quspin](https://github.com/weinbe58/QuSpin#installation)
- Seaborn (install using pip or conda)

# To run #

- Clone or download this repository. 
- Make sure you have a directory called ```data/``` where you want to run the code
- Then run the following command :
```
python LZ_sim_anneal.py
```
This will run the code with the default parameters. You can also try the following:
```
python LZ_sim_anneal.py 8 -2. 2. 30 20 bang-bang8 out.txt 0.05 100 False
```
This will run simulated annealing for 8 sites, from hx=-2. to h=2.0 state with 30 quenches and 20 time setps. The output file is out.txt and dt=0.05, with 100 restart. Verbose is set to false.

Finally, for more details on the meaning of those parameters (from the command line) use:
```
python LZ_sim_anneal.py -h
```
More details are given in the file ```LZ_sim_anneal.py```

