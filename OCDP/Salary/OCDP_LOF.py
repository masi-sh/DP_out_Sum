# This code investigates whether removing a datapoint affects the list of valid contexts for a particular outlier or not. 
from __future__ import division
import sys
#import gzip
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

random.seed(int(sys.argv[1])) 
query_num = int(sys.argv[1])
df = pd.read_csv("~/DP_out_Sum/Grubbs/ToyData.csv")
Ref_file = '/home/sm2shafi/DP_out_Sum/LOF/LOFRef.txt'
Query_file = '/home/sm2shafi/DP_out_Sum/LOF/TLQueries.csv'
Queries = pd.read_csv(Query_file)
# Check if the next line works
Queried_ID = int(Queries.iloc[query_num,1])
OutFile = 'OCDPMatch_L_D10.txt'
MatchFile = 'Sal_Match_L.txt'
NMatchFile = 'Sal_NMatch_L.txt'
NumofNeighbors = 50
DropThr = 10

def org_ctx(Ref_file, Queried_ID):
	FirAtt_lst = df['Job Title'].unique()
        SecAtt_lst = df['Employer'].unique()
        ThrAtt_lst = df['Calendar Year'].unique()
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
					o_ctx_db  = df.loc[df['Job Title'].isin(FirAtt_Sprset[int(ctx[0])]) &\
							   df['Employer'].isin(SecAtt_Sprset[int(ctx[1])]) & df['Calendar Year'].isin(ThrAtt_Sprset[int(ctx[2])])]
					o_ctx_shape.append(o_ctx_db.shape[0])	

	f.close()
        o_ctx = sorted(o_ctx)
	print 'size of o_ctx is:', len(o_ctx)
	return o_ctx, o_ctx_shape;
        
def neighbor_ctx(df, ndf, Queried_ID):
	FirAtt_lst = df['Job Title'].unique()
  	SecAtt_lst = df['Employer'].unique()
  	ThrAtt_lst = df['Calendar Year'].unique()
  	# Supersets for each attribute
  	FirAtt_Sprset = sum(map(lambda r: list(combinations(FirAtt_lst[0:], r)), range(1, len(FirAtt_lst[0:])+1)), [])
  	SecAtt_Sprset = sum(map(lambda r: list(combinations(SecAtt_lst[0:], r)), range(1, len(SecAtt_lst[0:])+1)), [])
  	ThrAtt_Sprset = sum(map(lambda r: list(combinations(ThrAtt_lst[0:], r)), range(1, len(ThrAtt_lst[0:])+1)), [])
  	ctx_count = 0
  	n_ctx = []
	n_ctx_shape =[]
  	for i in range (0, len(FirAtt_Sprset)):
    		for j in range (0, len(SecAtt_Sprset)):
      			for z in range(0, len(ThrAtt_Sprset)):
                		ctx_count+=1
                		Ctx  = ndf.loc[ndf['Job Title'].isin(FirAtt_Sprset[i]) & ndf['Employer'].isin(SecAtt_Sprset[j]) &\
					       ndf['Calendar Year'].isin(ThrAtt_Sprset[z])]
                		Sal_list = []
                		ID_list  = []
                		if (Ctx.shape[0] > 20):
       		        		for row in range(Ctx.shape[0]):
                    				Sal_list.append(Ctx.iloc[row,8])
                    				ID_list.append(Ctx.iloc[row,1])
        		    		Sal_arr= np.array(Sal_list)
        		    		clf = LocalOutlierFactor(n_neighbors=20)
                			Sal_outliers = clf.fit_predict(Sal_arr.reshape(-1,1))
                			for outlier_finder in range(0, len(ID_list)):
                  				if ((Sal_outliers[outlier_finder]==-1) and (ID_list[outlier_finder] == Queried_ID)): 
                    					n_ctx.append([i,j,z])     
							n_ctx_shape.append(Ctx.shape[0])
  	n_ctx = sorted(n_ctx)
        print 'size of n_ctx is:', len(n_ctx)
	return n_ctx, n_ctx_shape;   
        
def neighbors_compare(o_ctx , n_ctx, match_num):
  	# Caution: the following considers the permutation as inequality, double check if n_ctx has the same order as o_ctx or sort first
  	if (np.array_equal(o_ctx,n_ctx)):
    		match_num+=1
  	return match_num;   

def writefinal(OutputFile, DataToWrite):
        ff = open(OutputFile,'a+')
        fcntl.flock(ff, fcntl.LOCK_EX)
	np.savetxt(ff, np.column_stack(DataToWrite), fmt=('%7.5f'), header = \
		   'LOF (1.5, n=20) on Murder dataset: Num_of_COE_Match and len(COE_org), ')
        fcntl.flock(ff, fcntl.LOCK_UN)
        ff.close()
        return;

t0 = time.time()  
o_ctx , o_ctx_shape = org_ctx(Ref_file, Queried_ID)
match_num = 0
for neighbor in range (0, NumofNeighbors):
  	ndf = pd.DataFrame()
	ndf = df
	randomlist = random.sample(range(0, len(ndf)), DropThr)
	ndf = ndf.drop(randomlist)
  	n_ctx, n_ctx_shape = neighbor_ctx(df, ndf, Queried_ID)
  	match_num = neighbors_compare(o_ctx , n_ctx, match_num)  
	print 'match_num is: ', match_num, 'for the neighbor number ', neighbor	
DataToWrite = [match_num, len(o_ctx)]
writefinal(OutFile, DataToWrite)
t1 = time.time()
runtime = str(int((t1-t0) / 3600)) + ' hours and ' + str(int(((t1-t0) % 3600)/60)) +' minutes and ' + str(((t1-t0) % 3600)%60) + ' seconds\n'
print '\n OCDPMatch runtime,for LOF and ' , NumofNeighbors, ' neighbors is: \n', runtime
