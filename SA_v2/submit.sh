#!/bin/bash -login
#$ -P fheating
#$ -N job_0
#$ -l h_rt=12:00:00
#$ -m n
~/.conda/envs/py35/bin/python ES_para.py ${L} $nstep $slice $Treal > log/out_$nstep.log