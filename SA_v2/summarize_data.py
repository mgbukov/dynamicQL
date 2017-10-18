import numpy as np
import os, sys
from collections import OrderedDict
import pickle


def convertstr(mystr, type_expect):
    mystr = mystr.replace('m','-').replace('p','.')
    return type_expect(mystr)

def parse_name(fname):

    param = {
        'L' : int,
		'dt' : float,
		'J' : float,
		'nStep': int,
		'hz' : float,
		'hxI' : float,
		'hxF' : float,
		'hxmax' : float,
		'hxmin' : float,
		'dh' : float,
		'nSample' : int,
		'nQuench' : int,
		'Ti' : float,
		'T' : float,
		'symm' : int,
		'outfile' : str,
		'verbose' : int,
		'task' : str,
		'root' : str,
		'fid_series': bool
    }

    eachoption = fname.strip('.pkl').split('_')
    option = OrderedDict()
    option['task'] = eachoption[0]
    for par in eachoption[1:]:
        k, v = par.split('=')
        option[k] = convertstr(v, param[k])

    return option

def main():

    argv = sys.argv[1:]
    #print(argv)
    os.system('rm .file_here.tmp')
    tmp = ''
    for a in argv:
        tmp+= '*' + a.replace('.','p').replace('-','m') + '*'
    cmd = 'ls *%s* > .file_here.tmp'%tmp
    os.system(cmd)

    fopen = open('.file_here.tmp','r') # read corresponding tags
    all_file = [f.strip('\n') for f in fopen]
    # count number of samples, spit out rough info
    # print out the important part of the tag
    for fname in all_file:
        #print(fname)
        info = parse_name(fname)
        fpickle = open(fname,'rb')
        if info['task'] != 'ES':
            _, data = pickle.load(fpickle)
        else:
            #print(pickle.load(fpickle))
            data = pickle.load(fpickle)

        if type(data) == list:
            val = (info['task'], info['T'], info['L'], info['nStep'], info['dt'], len(data), len(data[0]))
            print("task: {0:<5s}T: {1:<6.2f}L: {2:<6d}nStep: {3:<6d}dt: {4:<8.4f}nSample: {5:<8d}nCol: {6:<6d}".format(*val))
    
        else:
            val = (info['task'], info['T'], info['L'], info['nStep'], info['dt'], data.shape[0], data.shape[1])
            print("task: {0:<5s}T: {1:<6.2f}L: {2:<6d}nStep: {3:<6d}dt: {4:<8.4f}nSample: {5:<8d}nCol: {6:<6d}".format(*val))

if __name__ =="__main__":
    main()