#!bin/bash

dt=0.02
for L in 14 #$(seq 2 2 4)
do

for nstep in $(seq 2 1 200) # of total ramp times
do
#dt=$(bc <<< "scale=8;$T/$nstep")
echo "#!/bin/bash -login" >> submission.sh
echo "#$ -P fheating" >> submission.sh #evolve is name of job
echo "#$ -N j_L${L}_${nstep}_${dt}" >> submission.sh #evolve is name of job
echo "#$ -l h_rt=36:00:00">> submission.sh #336
#echo "#$ -pe omp 4">> submission.sh #request more processors
echo "#$ -m ae" >> submission.sh #a (abort) e(exit)
#echo "#$-l mem_per_core=4G" >> submission.sh # more memory
#echo "#$ -M agrday@bu.edu" >> submission.sh
echo "#$ -m n" >> submission.sh # disable emails
#echo "source activate py35" >> submission.sh # my conda env
#sleep 2.0 # wait for 2 secs
echo "~/.conda/envs/py35/bin/python LZ_sim_anneal.py ${L} -2. 2. 0 $nstep bang-bang8 auto $dt 1000 False > log/out_${L}_$nstep.log" >> submission.sh

	        
qsub submission.sh
rm submission.sh
	        
sleep 0.1 #wait for half a second

done
done

