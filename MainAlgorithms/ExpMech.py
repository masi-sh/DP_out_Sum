
from __future__ import division
import matplotlib
from mpmath import mp
matplotlib.use('Agg')
import sys
#import gzip
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
import time
import fcntl
import random
import csv
import math

Query_num = int(sys.argv[1])
Query_file = '/home/sm2shafi/DP_out_Sum/MainAlgorithms/Queries.csv'
Queries = pd.read_csv(Query_file, 'rt', delimiter=',' , engine = 'python')
Ref_file = '/home/sm2shafi/Reffile.txt'
Store_file = 'Exp.dat'

Queried_ID = Queries.iloc[Query_num]['Outlier']
print '\n\n Outlier\'s ID in the original context is: ', Queried_ID
# finding maximal context's size for queried_ID
max_ctx = Queries.iloc[Query_num]['Max']
print '\nmaximal context has the population :\n', max_ctx

Data_to_write = []
t0 = time.time()
# Exp Mech on Queried_ID      
def Exp_Mech(Ref_file, Queried_ID, max_ctx):
	out_size    = 0
	size        = 0
	with open(Ref_file,'rt') as f:
		Exp_Can = []
		for num, line in enumerate(f, 1):
			if line.split(' ')[0].strip()=="Matching":
				size = int((line.split(' '))[5].strip(':\n'))
			elif line.strip().startswith("ID"):
				if line.split(' ')[3].strip('#')==str(Queried_ID):
					out_size = size
					Exp_Can.append(out_size)
					print 'Exp_Can is: ', Exp_Can

        f.close()
	print 'Running Exp over candidates...'
	elements = [elem for elem in range(len(Exp_Can))]
	probabilities =[]
	for prob in Exp_Can:
	   probabilities.append(prob/(sum(Exp_Can)))
	Res = np.random.choice(elements, 1, p = probabilities)
	Exp = Res[0]/max_ctx
	return Exp;

Data_to_write = Exp_Mech(Ref_file, Queried_ID, max_ctx)
t1 = time.time()
runtime = str(int((t1-t0) / 3600)) + ' hours and ' + str(int(((t1-t0) % 3600)/60)) + \
	' minutes and ' + str(((t1-t0) % 3600)%60) + ' seconds\n'

ff = open(Store_file,'a+')
fcntl.flock(ff, fcntl.LOCK_EX)
np.savetxt(ff, np.column_stack(Data_to_write), fmt=('%7.5f'), header = randomness+ ' Generates outlier , ' + ID + ', \
Exp alg. takes' + runtime)
fcntl.flock(ff, fcntl.LOCK_UN)
ff.close()
