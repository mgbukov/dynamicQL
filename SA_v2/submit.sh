#!/bin/bash -login
#$ -P fheating
#$ -N job_39
#$ -l h_rt=12:00:00
#$ -m n
#$ -m ae
~/.conda/envs/py35/bin/python main.py n_step=40 Ti=0.29
