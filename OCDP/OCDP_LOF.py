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

query_num = int(sys.argv[1])
df = pd.read_csv("~/DP_out_Sum/Grubbs/ToyData.csv")
Ref_file = '/home/sm2shafi/DP_out_Sum/LOF/LOFRef.txt'
Query_file = '/home/sm2shafi/DP_out_Sum/LOF/TLQueries.csv'
Queries = pd.read_csv(Query_file)
# Check if the next line works
Queried_ID = int(Queries.iloc[query_num,0])
OutFile = 'OCDPMatch.txt'

def org_ctx(Ref_file, Queried_ID):
	with open(Ref_file,'rt') as f:
		o_ctx = []
    		for num, line in enumerate(f, 1):
      			ctx = line[1:-2].split(',')
			# Double check, chnaged for outliers in range(len(ctx)) to for outliers in range(4, len(ctx))
      			for outliers in range(4, len(ctx)):
				if int(ctx[outliers])==Queried_ID:
          				# Double check if this holds: [ctx[0],ctx[1],ctx[2]] = [i, j, z]
          				o_ctx.append(ctx[0]+ 1000*ctx[1] + 1000000*ctx[2])		
	f.close()
	return o_ctx;
        
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
  	for i in range (0, len(FirAtt_Sprset)):
    		for j in range (0, len(SecAtt_Sprset)):
      			for z in range(0, len(ThrAtt_Sprset)):
                		ctx_count+=1
                		print 'count is:', ctx_count #, ' The percentage done: %', ctx_count//(2**14) 
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
                    					n_ctx.append(i+ 1000*j + 1000000*z)
  	return n_ctx;   
        
def neighbors_compare(o_ctx , n_ctx, match_num):
  	# Caution: the following considers the permutation as inequality, double check if n_ctx has the same order as o_ctx or sort first
  	if (np.array_equal(o_ctx,n_ctx)):
    		match_num+=1
  	return match_num;   

def writefinal(OutFile, match_num):
        ff = open(OutFile,'a+')
        fcntl.flock(ff, fcntl.LOCK_EX)
        ff.write(str(match_num)+'\n')
        fcntl.flock(ff, fcntl.LOCK_UN)
        ff.close()
        return;

t0 = time.time()  
o_ctx = org_ctx(Ref_file, Queried_ID)
match_num = 0
for neighbor in range (0, 10):
  	ndf = pd.DataFrame()
  	neighbor_rnd = np.random.randint(len(df)-1)
  	ndf = df.drop(neighbor_rnd)
  	n_ctx = neighbor_ctx(df, ndf, Queried_ID)
  	match_num = neighbors_compare(o_ctx , n_ctx, match_num)  
	print 'match_num is: ', match_num, 'for the neighbor number ', neighbor	
writefinal(OutFile, match_num)
t1 = time.time()
print '\n\nThe required time for running the program is:',  t1-t0
     
