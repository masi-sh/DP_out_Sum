# This code investigates whether removing a datapoint affects the list of valid contexts for a particular outlier or not. 
# HIST is used as the outlier detection algorithm, on the Murder dataset
from __future__ import division
import sys
import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter
from sklearn.neighbors import LocalOutlierFactor
import time
import fcntl
import random
import csv
import math

query_num = int(sys.argv[1])
df = pd.read_csv("~/DP_out_Sum/dataset/MurderData_28.csv")
Ref_file = '/home/sm2shafi/DP_out_Sum/Murder/HIST/MHistRef_28.txt'
Query_file = '/home/sm2shafi/DP_out_Sum/Murder/HIST/MHQueries_28.csv'
Queries = pd.read_csv(Query_file)
# Check if the next line works
Queried_ID = int(Queries.iloc[query_num,1])
OutFile = 'M_OCDPMatch_H.txt'
NumofNeighbors = 50
DropThr = 10

def org_ctx(Ref_file, Queried_ID):
	with open(Ref_file,'rt') as f:
		o_ctx = []
    		for num, line in enumerate(f, 1):
      			ctx = line[1:-2].split(',')
			# Double check, chnaged for outliers in range(len(ctx)) to for outliers in range(4, len(ctx))
      			for outliers in range(4, len(ctx)):
				if int(ctx[outliers])==Queried_ID:
          				# Double check if this holds: [ctx[0],ctx[1],ctx[2]] = [i, j, z]
                                        o_ctx.append([int(ctx[0]), int(ctx[1]), int(ctx[2])])
	f.close()
        o_ctx = sorted(o_ctx)
        print 'size of o_ctx is:', len(o_ctx)	
	return o_ctx;
        
def neighbor_ctx(df, ndf, Queried_ID):
	FirAtt_lst = df['Weapon'].unique()
  	SecAtt_lst = df['State'].unique()
  	ThrAtt_lst = df['AgencyType'].unique()
  	# Supersets for each attribute
  	FirAtt_Sprset = sum(map(lambda r: list(combinations(FirAtt_lst[0:], r)), range(1, len(FirAtt_lst[0:])+1)), [])
  	SecAtt_Sprset = sum(map(lambda r: list(combinations(SecAtt_lst[0:], r)), range(1, len(SecAtt_lst[0:])+1)), [])
  	ThrAtt_Sprset = sum(map(lambda r: list(combinations(ThrAtt_lst[0:], r)), range(1, len(ThrAtt_lst[0:])+1)), [])
  	ctx_count = 0
  	n_ctx = []
  	for i in range (0, len(FirAtt_Sprset)):
    		for j in range (0, len(SecAtt_Sprset)):
      			for z in range(0, len(ThrAtt_Sprset)):
                		ctx_count+=1
                		print 'count is:', ctx_count #, ' The percentage done: %', ctx_count//(2**14) 
				Sal_bin    = []
                		Ctx  = ndf.loc[ndf['Weapon'].isin(FirAtt_Sprset[i]) & ndf['State'].isin(SecAtt_Sprset[j]) &\
					       ndf['AgencyType'].isin(ThrAtt_Sprset[z])]
				if (Ctx.shape[0] > 20):
                   			Salary = Ctx['VictimAge']
                        		IDs    = Ctx['Record ID']
                        		histi  = np.histogram(Salary.values, bins=int(np.sqrt(len(Salary.values))), density=False)
                        		bin_width = histi[1][1] - histi[1][0]
                        		for Sal_freq in range(len(histi[0])):
                                		if histi[0][Sal_freq] <= 0.0025*len(Ctx['VictimAge']):
                                        		Sal_bin.append(histi[1][Sal_freq])
                        		for Sal_idx in range(len(Salary.values)):
                                		if ((len(filter(lambda x : x <= Salary.values[Sal_idx] < x+bin_width , Sal_bin)) > 0) and (IDs.values[Sal_idx]==Queried_ID)):
                                                                n_ctx.append([i,j,z])
        n_ctx = sorted(n_ctx)
        print 'size of n_ctx is:', len(n_ctx)
	return n_ctx;   
        
def neighbors_compare(o_ctx , n_ctx, match_num):
  	# Caution: the following considers the permutation as inequality, double check if n_ctx has the same order as o_ctx or sort first
  	if (np.array_equal(o_ctx,n_ctx)):
    		match_num+=1
  	return match_num;  

def writefinal(OutputFile, DataToWrite):
        ff = open(OutputFile,'a+')
        fcntl.flock(ff, fcntl.LOCK_EX)
	np.savetxt(ff, np.column_stack(DataToWrite), fmt=('%7.5f'), header = \
		   'Hist on Murder dataset: Num_of_COE_Match and len(COE_org), ')
        fcntl.flock(ff, fcntl.LOCK_UN)
        ff.close()
        return;

t0 = time.time()  
o_ctx = org_ctx(Ref_file, Queried_ID)
match_num = 0
for neighbor in range (0, NumofNeighbors):
  	ndf = pd.DataFrame()
	ndf = df
	randomlist = random.sample(range(0, len(ndf)), DropThr)
	ndf = ndf.drop(randomlist)
  	n_ctx = neighbor_ctx(df, ndf, Queried_ID)
  	match_num = neighbors_compare(o_ctx , n_ctx, match_num)  
	print 'match_num is: ', match_num, 'for the neighbor number ', neighbor	
DataToWrite = [match_num, len(o_ctx)]
writefinal(OutFile, DataToWrite)
t1 = time.time()
runtime = str(int((t1-t0) / 3600)) + ' hours and ' + str(int(((t1-t0) % 3600)/60)) +' minutes and ' + str(((t1-t0) % 3600)%60) + ' seconds\n'
print '\n OCDPMatch runtime,for LOF and ' , NumofNeighbors, ' neighbors is: \n', runtime
