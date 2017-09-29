#!/bin/bash -login
#$ -P fheating
#$ -N job_SD_0
#$ -l h_rt=12:00:00
#$ -m ae
#$ -m n
~/.conda/envs/py35/bin/python main_Ising.py dt=0.01 n_quench=10000 L=2 outfile=auto verbose=1 hx_max=4.0 J=1.0 hx_f=2.0 task=ES n_sample=1000 Ti=-1.0 dh=8.0 hx_min=-4.0 hx_i=-2.0 symmetrize=0 T=1.0 hz=1.0 n_step=10

main.py dt=0.02 n_quench=10000 L=2 outfile=auto verbose=1 hx_max=4.0 J=1.0 hx_f=2.0 task=SD n_sample=1000 Ti=-1.0 dh=8.0 hx_min=-4.0 hx_i=-2.0 symmetrize=0 T=1.0 hz=1.0 n_step=270




main.py dt=0.01 n_step=260 L=2 J=1.0 hz=1.0 hx_i=-2.0 hx_f=2.0 hx_min=-4.0 hx_max=4.0 dh=8.0 task=SD n_sample=100 n_quench=10000 outfile=auto verbose=1 Ti=-1.0 T=1.0 symmetrize=0

main.py dt=0.01 n_step=260 L=2 J=1.0 hz=1.0 hx_i=-2.0 hx_f=2.0 hx_min=-8.0 hx_max=8.0 dh=16.0 task=SD n_sample=100 n_quench=10000 outfile=auto verbose=1 Ti=-1.0 T=1.0 symmetrize=0
