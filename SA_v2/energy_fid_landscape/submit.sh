#!/bin/bash -login
#$ -P fheating"
#$ -N density_L6
#$ -l h_rt=4:00:00
#$ -m ae
#$ -m n
~/.conda/envs/py35/bin/python energy_vs_fid.py


     



