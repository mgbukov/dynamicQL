#!bin/bash

for i in $(seq 1 1 11) # frequencies
do

	for j in $(seq 1 1 11) # g's
	do

		echo "#!/bin/bash -login" >> submission.sh
        echo "#$ -P fheating" >> submission.sh #evolve is name of job
        echo "#$ -N evolve_${j}_${i}" >> submission.sh #evolve is name of job
        echo "#$ -l h_rt=72:00:00">> submission.sh #336
        echo "#$ -pe omp 4">> submission.sh #request more processors
        echo "#$ -m ae" >> submission.sh #a (abort) e(exit)
        echo "#$-l mem_per_core=4G" >> submission.sh #memory
        #echo "#$ -M mbukov@bu.edu" >> submission.sh
        echo "#$ -m n" >> submission.sh # disable emails
        echo "source activate ED" >> submission.sh
        echo "matlab -nodesktop -nodisplay -singleCompThread -r 'tubes_2D(${j},${i}); quit;' " >> submission.sh
        
        qsub submission.sh
        rm submission.sh
        
        sleep 1.0 #wait for half a second

	done


done