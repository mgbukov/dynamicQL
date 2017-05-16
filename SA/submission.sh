#!/bin/bash -login
#$ -P fheating
#$ -N j_L_6_22
#$ -l h_rt=12:00:00
#$ -m ae
#$ -m n
~/.conda/envs/py35/bin/python ES_para.py 6 22 0 5.8 > log/out_22.log
#!/bin/bash -login
#$ -P fheating
#$ -N j_L_6_22
#$ -l h_rt=12:00:00
#$ -m ae
#$ -m n
~/.conda/envs/py35/bin/python ES_para.py 6 22 1 5.8 > log/out_22.log
#!/bin/bash -login
#$ -P fheating
#$ -N j_L_6_22
#$ -l h_rt=12:00:00
#$ -m ae
#$ -m n
~/.conda/envs/py35/bin/python ES_para.py 6 22 2 5.8 > log/out_22.log
