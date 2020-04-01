# This code investigates whether removing a datapoint affects the list of valid contexts for a particular outlier or not. 
# Grubbs is used as the outlier detection algorithm, on the Murder dataset
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
import hashlib
from outliers import smirnov_grubbs as grubbs

random.seed(int(sys.argv[1])) 
query_num = int(sys.argv[1])
df = pd.read_csv("~/DP_out_Sum/dataset/MurderData_28.csv")
Ref_file = '/home/sm2shafi/DP_out_Sum/Murder/Grubbs/MGrubbsRef_28.txt'
Query_file = '/home/sm2shafi/DP_out_Sum/Murder/Grubbs/MGQueries_28.csv'
Queries = pd.read_csv(Query_file)
# Check if the next line works
Queried_ID = int(Queries.iloc[query_num,1])
OutFile  = 'M_OCDPMatch_G.txt'
MatchFile = 'Mur_Match_G.txt'
NMatchFile = 'Mur_NMatch_G.txt'
NumofNeighbors = 50
DropThr = 10

def org_ctx(df, Ref_file, Queried_ID):
	FirAtt_lst = df['Weapon'].unique()
  	SecAtt_lst = df['State'].unique()
  	ThrAtt_lst = df['AgencyType'].unique()
  	# Supersets for each attribute
  	FirAtt_Sprset = sum(map(lambda r: list(combinations(FirAtt_lst[0:], r)), range(1, len(FirAtt_lst[0:])+1)), [])
  	SecAtt_Sprset = sum(map(lambda r: list(combinations(SecAtt_lst[0:], r)), range(1, len(SecAtt_lst[0:])+1)), [])
  	ThrAtt_Sprset = sum(map(lambda r: list(combinations(ThrAtt_lst[0:], r)), range(1, len(ThrAtt_lst[0:])+1)), [])
	with open(Ref_file,'rt') as f:
		o_ctx = []
		o_ctx_shape = []
    		for num, line in enumerate(f, 1):
      			ctx = line[1:-2].split(',')
      			for outliers in range(4, len(ctx)):
				if int(ctx[outliers])==Queried_ID:
          				o_ctx.append([int(ctx[0]), int(ctx[1]), int(ctx[2])])
					o_ctx_db  = df.loc[df['Weapon'].isin(FirAtt_Sprset[int(ctx[0])]) &\
							   df['State'].isin(SecAtt_Sprset[int(ctx[1])]) & df['AgencyType'].isin(ThrAtt_Sprset[int(ctx[2])])]
					o_ctx_shape.append(o_ctx_db.shape[0])				
	f.close()
	#o_ctx = sorted(o_ctx)
	print 'size of o_ctx is:', len(o_ctx)
	return o_ctx, o_ctx_shape;
        
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
	n_ctx_shape = []
  	for i in range (0, len(FirAtt_Sprset)):
    		for j in range (0, len(SecAtt_Sprset)):
      			for z in range(0, len(ThrAtt_Sprset)):
                		ctx_count+=1
                		Ctx  = ndf.loc[ndf['Weapon'].isin(FirAtt_Sprset[i]) & ndf['State'].isin(SecAtt_Sprset[j]) &\
					       ndf['AgencyType'].isin(ThrAtt_Sprset[z])]
				if (Ctx.shape[0] > 20):
					Salary = Ctx['VictimAge']
        	        		IDs    = Ctx['Record ID']
                			grubbs_result = grubbs.max_test_indices(Salary, alpha=0.05)
                			if grubbs_result:
						for GOutlier in grubbs_result:
                                			if (IDs.values[GOutlier]==Queried_ID):
								n_ctx.append([i,j,z])
								n_ctx_shape.append(Ctx.shape[0])
  	#n_ctx = sorted(n_ctx)
        print 'size of n_ctx is:', len(n_ctx)
	return n_ctx, n_ctx_shape;   
        
def neighbors_compare(o_ctx , n_ctx, match_num, o_ctx_shape, n_ctx_shape):
  	if (np.array_equal(sorted(o_ctx),sorted(n_ctx))):
    		match_num+=1
		writefinal(MatchFile, o_ctx[:])
		writefinal(MatchFile, o_ctx_shape[:])
		writefinal(MatchFile, n_ctx[:])
		writefinal(MatchFile, n_ctx_shape[:])
	else:	
		writefinal(NMatchFile,'#' + 'Queried_ID is: ' + str(Queried_ID) + ', removed records are: ' + str(randomlist))
		writefinal(NMatchFile, o_ctx[:])
		writefinal(NMatchFile, o_ctx_shape[:])
		writefinal(NMatchFile, n_ctx[:])
		writefinal(NMatchFile, n_ctx_shape[:])
  	return match_num;   

def writefinal(OutputFile, DataToWrite):
        ff = open(OutputFile,'a+')
        fcntl.flock(ff, fcntl.LOCK_EX)
	np.savetxt(ff, np.column_stack(DataToWrite), fmt=('%7.5f'), header = \
		   'Grubbs on Murder dataset: Num_of_COE_Match and len(COE_org), ')
        fcntl.flock(ff, fcntl.LOCK_UN)
        ff.close()
        return;

t0 = time.time()  
o_ctx, o_ctx_shape = org_ctx(df, Ref_file, Queried_ID)
match_num = 0
for neighbor in range (0, NumofNeighbors):
  	ndf = pd.DataFrame()
	ndf = df
	randomlist = random.sample(range(0, len(ndf)), DropThr)
	ndf = ndf.drop(randomlist)
  	n_ctx, n_ctx_shape = neighbor_ctx(df, ndf, Queried_ID)
  	match_num = neighbors_compare(o_ctx , n_ctx, match_num, o_ctx_shape, n_ctx_shape)  
	print 'match_num is: ', match_num, 'for the neighbor number ', neighbor
DataToWrite = [match_num, len(o_ctx)]
writefinal(OutFile, DataToWrite)
t1 = time.time()
runtime = str(int((t1-t0) / 3600)) + ' hours and ' + str(int(((t1-t0) % 3600)/60)) +' minutes and ' + str(((t1-t0) % 3600)%60) + ' seconds\n'
print '\n OCDPMatch runtime,for Grubbs and ' , NumofNeighbors, ' neighbors is: \n', runtime
