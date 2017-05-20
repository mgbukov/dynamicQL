### python scripts for submitting job on scc using qsub
import os, subprocess, time
from numpy import arange # for iterating over real values

parameters = {
    'project': 'fheating',
    'job_name': 'job_SD_%i',
    'walltime': '24:00:00',
    'command' : '~/.conda/envs/py35/bin/python main.py',
    'arguments' : [],# fixed parameters
    'loop' : [['n_step',[4,6,8,10,12,14,16,18,20,22,24,26,28]],['T',[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]]] # looping parameters
}

###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################

def main():
    global submit_count
    submit_count = 0
    submit(parameters,exe=True)

def write_header(file, parameters):
    global submit_count
    target = open(file, 'w')
    target.write('#!/bin/bash -login')
    target.write('\n')
    target.write("#$ -P %s" % parameters['project'])
    target.write('\n')
    target.write("#$ -N %s" % (parameters['job_name']) % submit_count) 
    target.write('\n')
    target.write("#$ -l h_rt=%s\n" % parameters['walltime'])
    target.write("#$ -m ae\n")
    target.write("#$ -m n\n")
    return target

def submit(parameters, file='submit.sh',exe = False):
    write_header(file, parameters)

    global submit_count
    for arg in parameters['arguments']:
        parameters['command']+=(" "+arg[0]+"="+str(arg[1]))

    if exe is True:
        os.system('rm %s'%file)
        n_loop = len(parameters['loop'])
        if n_loop == 0:
            target = write_header(file, parameters)
            target.write(parameters['command']+"\n")
            target.close()
            os.system('qsub %s'%file)
            os.system('rm %s'%file)
            submit_count += 1

        elif n_loop == 1:
            loop_1 = parameters['loop'][0]
            tag, iterable = (loop_1[0],loop_1[1])
            for value in iterable:    
                target = write_header(file, parameters)
                target.write(parameters['command']+(" "+tag+"="+str(value)+'\n'))
                target.close()
                os.system('qsub %s'%file)
                os.system('rm %s'%file)
                time.sleep(0.1)
                submit_count+=1

        elif n_loop == 2:
            loop_1 = parameters['loop'][0]
            loop_2 = parameters['loop'][1]
            tag_1, iterable_1 = (loop_1[0],loop_1[1])
            tag_2, iterable_2 = (loop_2[0],loop_2[1])
            for value_1 in iterable_1:
                for value_2 in iterable_2:    
                    target = write_header(file, parameters)
                    target.write(parameters['command']+(" "+tag_1+"="+str(value_1)+" "+tag_2+"="+str(value_2)+'\n'))
                    target.close()
                    os.system('qsub %s'%file)
                    os.system('rm %s'%file)
                    time.sleep(0.1)
                    submit_count+=1

if __name__=='__main__':
    main()

