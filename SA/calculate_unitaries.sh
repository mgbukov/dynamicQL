#!bin/bash

for dt in 0.02 0.01 0.005 0.0025
do

for L in $(seq 2 2 14)
do

python LZ_sim_anneal.py $L -2. 2. 0 2 bang-bang8 auto ${dt} 1 True

done
done



