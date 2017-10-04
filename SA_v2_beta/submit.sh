#!/bin/bash -login
#$ -P fheating
#$ -N job_SD_0
#$ -l h_rt=12:00:00
#$ -m ae
#$ -m n
~/.conda/envs/py35/bin/python main.py dt=0.01 norm=GS hx_f=2.0 T=1.0 n_sample=1000 outfile=auto hx_i=-2.0 verbose=1 hz=1.0 hx_min=-4.0 L=2 symmetrize=0 task=SD J=1.0 Ti=-1.0 n_quench=10000 hx_max=4.0 dh=8.0 n_step=10
