#!bin/bash

L=6
nstep=26

for T in $(seq 1 1 60)
do
	for slice in $(seq 0 1 31) # of total ramp times
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
=======
dt=0.0025

for L in 2 #$(seq 2 2 10)
do

for nstep in 310  #$(seq 10 10 200) # of total ramp times
do
#dt=$(bc <<< "scale=8;$T/$nstep")
echo "#!/bin/bash -login" >> submission.sh
echo "#$ -P fheating" >> submission.sh #evolve is name of job
echo "#$ -N j_L${L}_${nstep}_${dt}" >> submission.sh #evolve is name of job
echo "#$ -l h_rt=12:00:00">> submission.sh #336
#echo "#$ -pe omp 4">> submission.sh #request more processors
echo "#$ -m ae" >> submission.sh #a (abort) e(exit)
#echo "#$-l mem_per_core=4G" >> submission.sh # more memory
#echo "#$ -M agrday@bu.edu" >> submission.sh
echo "#$ -m n" >> submission.sh # disable emails
#echo "source activate py35" >> submission.sh # my conda env
#sleep 2.0 # wait for 2 secs
echo "~/.conda/envs/py35/bin/python LZ_sim_anneal.py ${L} -2. 2. 0 $nstep bang-bang8 auto $dt 1000 False True > log/out_${L}_$nstep.log" >> submission.sh
>>>>>>> 0bfd8b12a2103a0cf899d171113ffc0ac160eeb1:SA/submit_scc_new.sh

		qsub submission.sh
		rm submission.sh
	        
<<<<<<< HEAD:SA/submit_scc.sh
		sleep 0.5 #wait for half a second
	done
=======
qsub submission.sh
rm submission.sh
	        
sleep 0.1 #wait for half a second

>>>>>>> 0bfd8b12a2103a0cf899d171113ffc0ac160eeb1:SA/submit_scc_new.sh
done
done

