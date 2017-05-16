#!bin/bash

L=6
nstep=22

for T in $(seq 58 1 58)
do
	for slice in $(seq 0 1 2) # of total ramp times
	do
		Treal=$(bc <<< "scale=8;$T*0.1")
		echo "#!/bin/bash -login" >> submission.sh
		echo "#$ -P fheating" >> submission.sh #evolve is name of job
		echo "#$ -N j_L_${L}_${nstep}" >> submission.sh #evolve is name of job
		echo "#$ -l h_rt=12:00:00">> submission.sh #336
		#echo "#$ -pe omp 4">> submission.sh #request more processors
		echo "#$ -m ae" >> submission.sh #a (abort) e(exit)
		#echo "#$-l mem_per_core=4G" >> submission.sh # more memory
		#echo "#$ -M agrday@bu.edu" >> submission.sh
		echo "#$ -m n" >> submission.sh # disable emails
		#echo "source activate py35" >> submission.sh # my conda env
		echo $Treal,$slice
		echo "~/.conda/envs/py35/bin/python ES_para.py ${L} $nstep $slice $Treal > log/out_$nstep.log" >> submission.sh

		#qsub submission.sh
		#rm submission.sh
	done
	        
sleep 0.1 #wait for half a second

done

