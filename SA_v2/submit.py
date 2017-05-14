### python scripts for submitting job on scc
import os,sys
import itertools
import time

parameters = {
    'project': 'fheating',
    'job_name': 'job_%i',
    'walltime': '12:00:00',
    'command' : '~/.conda/envs/py35/bin/python main.py %s > out.log',
    'loop' : [['n_step',range(10,41,10)]]
}

def main():
    global submit_count
    submit_count = 0
    submit(parameters)


def write_header(file, parameters):
    global submit_count
    target = open(file, 'w')
    target.write('#!/bin/bash -login')
    target.write('\n')
    target.write("#$ -P %s" % parameters['project'])
    target.write('\n')
    target.write("#$ -N %s" % (parameters['job_name']) % submit_count) 
    target.write('\n')
    target.write("#$ -l h_rt=%s" % parameters['walltime'])
    target.write("#$ -m n\n#$ -m ae\n")
    return target



def submit(parameters, file='submit.sh',exe = False):
    global submit_count
    if exe is True:
        n_loop = len(parameters['loop'])
        if n_loop == 0:
            target = write_header(file, parameters)
            target.write(parameters['command']%"")
            os.system('qsub %s'%file)
            os.system('rm %s'%file)
            submit_count += 1
        elif n_loop == 1:
            target = write_header(file, parameters)
            loop_1 = parameters['loop'][0]
            tag, iterable = (loop_1[0],loop_1[1])
            for value in iterable:
                  target.write(parameters['command']%(tag+"="+str(value))
                  os.system('qsub %s'%file)
                  os.system('rm %s'%file)
                  time.sleep(0.1)
                  submit_count+=1
        elif n_loop == 2:
            loop_1 = parameters['loop'][0]
            loop_2 = parameters['loop'][1]


if __name__=='__main__':
    main()

