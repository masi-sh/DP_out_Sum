import sys
import pandas as pd
import numpy as np
import cufflinks as cf
cf.go_offline()
from itertools import combinations
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
import time
import fcntl
import random
import csv
import math
import hashlib

random.seed(50*int(sys.argv[1]))
query_num = int(sys.argv[1])
Query_file = '~/DP_out_Sum/dataset/RndQueries.csv'
Queries = pd.read_csv(Query_file)
df2 = pd.read_csv("~/DP_out_Sum/dataset/FilteredData.csv")
Ref_file = '~/DP_out_Sum/Other/output.txt'

def maxctx(Ref_file, Queried_ID):
	max  = 0
	size = 0
	outlier_ctr = 0
	with open('output.txt','rt') as f:
        	for num, line in enumerate(f, 1):
			ctx = line[1:-2].split(',')
                	size = int((line[1:-1].split(','))[3])
			for outliers in range(len(ctx)):
				if int(ctx[outliers])==ID:
   	        			outlier_ctr += 1
					if (max < size):
						max = size
					break				
	f.close()
	return max, outlier_ctr;

def hash_calc(i, j, z, ID):
        hash_value = hashlib.md5(str(i+1000*j+1000000*z)+str(ID))
        hash_hex = hash_value.hexdigest()
        #:as_int = int(hash_hex[30:32],16)
        #return (as_int%128==0);
	return (hash_hex[30:32] == '80' or hash_hex[30:32] == '00');

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
	outliers.append([i, j, z, Ctx.shape[0]])
        if (Ctx.shape[0]>20):
        	for row in range(Ctx.shape[0]):
                	ID = Ctx.iloc[row, 0]
                        if hash_calc(i, j, z, ID):
				outliers.append([i, j, z, Ctx.shape[0], ID])
				break
		else:
            		continue
        	break
	max_ctx, count = 0,501
	#max_ctx, count = maxctx(Ref_file, ID)
	if (max_ctx !=0 and count>500):
		print 'count>500!'
		break
		
with open(Query_file, 'ab') as csvfile:
       	writer = csv.writer(csvfile)
       	fcntl.flock(csvfile, fcntl.LOCK_EX)
       	writer.writerow([query_num, ID, max_ctx, str(i+1000*j+1000000*z)])
       	fcntl.flock(csvfile, fcntl.LOCK_UN)
csvfile.close()
