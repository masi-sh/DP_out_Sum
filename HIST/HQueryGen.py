import sys
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
import hashlib
from outliers import smirnov_grubbs as grubbs

random.seed(50*int(sys.argv[1]))
query_num = int(sys.argv[1])
df2 = pd.read_csv("~/DP_out_Sum/Grubbs/ToyData.csv")
Query_file = '/home/sm2shafi/DP_out_Sum/HIST/THQueries.csv'
Queries = pd.read_csv(Query_file)
Ref_file = '/home/sm2shafi/DP_out_Sum/HIST/HistRef.txt'

def maxctx(Ref_file, Queried_ID):
	max  = 0
	size = 0
	outlier_ctr = 0
	with open(Ref_file,'rt') as f:
        	for num, line in enumerate(f, 1):
			ctx = line[1:-2].split(',')
                	size = int(ctx[3])
			for outliers in range(len(ctx)):
				if int(ctx[outliers])==ID:
   	        			outlier_ctr += 1
					if (max < size):
						max = size
					break				
	f.close()
	return max, outlier_ctr;

FirAtt_lst = df2['Job Title'].unique()
SecAtt_lst = df2['Employer'].unique()
ThrAtt_lst = df2['Calendar Year'].unique()

# Supersets for each attribute
FirAtt_Sprset = sum(map(lambda r: list(combinations(FirAtt_lst[0:], r)), range(1, len(FirAtt_lst[0:])+1)), [])
SecAtt_Sprset = sum(map(lambda r: list(combinations(SecAtt_lst[0:], r)), range(1, len(SecAtt_lst[0:])+1)), [])
ThrAtt_Sprset = sum(map(lambda r: list(combinations(ThrAtt_lst[0:], r)), range(1, len(ThrAtt_lst[0:])+1)), [])

outliers = []
while True:
	i = np.random.randint(len(FirAtt_Sprset)-1)
	j = np.random.randint(len(SecAtt_Sprset)-1)
	z = np.random.randint(len(ThrAtt_Sprset)-1)
	Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_Sprset[i]) & df2['Employer'].isin(SecAtt_Sprset[j]) &\
		       df2['Calendar Year'].isin(ThrAtt_Sprset[z])]
	#outliers.append([i, j, z, Ctx.shape[0]])
	Sal_bin = []
        if (Ctx.shape[0]>20):
		Salary = Ctx['Salary Paid']
		IDs = Ctx['Unnamed: 0.1']
		histi  = np.histogram(Salary.values, bins=int(np.sqrt(len(Salary.values))), density=False)
                bin_width = histi[1][1] - histi[1][0]
                for Sal_freq in range(len(histi[0])):
			if histi[0][Sal_freq] <= 0.0025*len(Ctx['Salary Paid']):
				Sal_bin.append(histi[1][Sal_freq])
		for Sal_idx in range(len(Salary.values)):
			if (len(filter(lambda x : x <= Salary.values[Sal_idx] < x+bin_width , Sal_bin)) > 0):
				ID = IDs.values[Sal_idx]
				break
		else:
			continue
		break
		
max_ctx, count = maxctx(Ref_file, ID)
if (max_ctx !=0 and count>500):
	print 'count>500!'
	with open(Query_file, 'ab') as csvfile:
       		writer = csv.writer(csvfile)
       		fcntl.flock(csvfile, fcntl.LOCK_EX)
       		writer.writerow([query_num, ID, max_ctx, str(i+1000*j+1000000*z)])
       		fcntl.flock(csvfile, fcntl.LOCK_UN)
	csvfile.close()
