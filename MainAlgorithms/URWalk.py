from __future__ import division
import matplotlib
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
# This file is filtered, no extra filtering required
df2 = pd.read_csv("~/DP_out_Sum/dataset/FilteredData.csv")
Query_file = '/home/sm2shafi/DP_out_Sum/MainAlgorithms/Queries.csv'
Queries = pd.read_csv(Query_file, 'rt', delimiter=',' , engine = 'python')
Store_file = 'URWalk.dat'

# Writing final data 
def writefinal(Data_to_write, randomness, runtime, ID, max_ctx):	
	ff = open(Store_file,'a+')
	fcntl.flock(ff, fcntl.LOCK_EX)
	np.savetxt(ff, np.column_stack(Data_to_write), fmt=('%7.5f'), header = 'URWalk for query number: '+ randomness +\
	'for outlier' + ID + 'with Ctx_max '+ str(max_ctx) + 'takes ' + runtime)	
	fcntl.flock(ff, fcntl.LOCK_UN)
	ff.close()
	return;

FirAtt_lst = df2['Job Title'].unique()
SecAtt_lst = df2['Employer'].unique()
ThrAtt_lst = df2['Calendar Year'].unique()
	
# Reading a Queried_ID from the list in the Queries file
Queried_ID = Queries.iloc[Query_num]['Outlier']
print '\n\n Outlier\'s ID in the original context is: ', Queried_ID
# finding maximal context's size for queried_ID
#max_ctx = Queries.iloc[Query_num]['Max']
#print '\nmaximal context has the population :\n', max_ctx

Org_Vec = np.zeros(len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst))
# polishing Ctx in Query_file and reading Org_Vec from it
Queries['Ctx'] = Queries['Ctx'].replace({'\n': ''}, regex=True)
Org_Str = Queries.iloc[Query_num]['Ctx'][1:-2].strip('[]').replace('.','').replace(' ', '')
for i in range(len(Org_Vec)):
	if (Org_Str[i] =='1'):
		Org_Vec[i] = 1
		
Orgn_Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(Org_Vec[0:len(FirAtt_lst)] == 1)].tolist()) &\
                    df2['Employer'].isin(SecAtt_lst[np.where(Org_Vec[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)] == 1)].tolist())  &\
                    df2['Calendar Year'].isin(ThrAtt_lst[np.where(Org_Vec[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)] == 1)].tolist())]
max_ctx = Orgn_Ctx.shape[0]
# Keeping attribute values in the original context, p =pr(1-->1) 
Flp_p        = 0.7
# Adding attribute values not in the original context, q =pr(0-->1)
Flp_q        = 0.4
# Flip the context, 100 times    
Epsilon = 0.001
Flp_lst	     = [[0, math.exp(Epsilon *(Orgn_Ctx.shape[0])), Orgn_Ctx.shape[0], Org_Vec]]
Data_to_write = []
t0 = time.time()
while len(Flp_lst)<50:
	print '\n len(Flp_lst) is = ', len(Flp_lst)
	# context separator scans all elements in the attribute lists to find where to apply p or q 
    	Vec_Flp = np.zeros(len(Org_Vec), dtype=np.int)
	for Ctx_sprt in range (0, len(Vec_Flp)):
        	if ((Flp_lst[len(Flp_lst)-1][3][Ctx_sprt]==1 and np.random.binomial(size=1, n=1, p= Flp_p)==1) or \
		    (Flp_lst[len(Flp_lst)-1][3][Ctx_sprt]==0 and np.random.binomial(size=1, n=1, p= Flp_q)==1)):
                	Vec_Flp[Ctx_sprt]=1
   	print '\n Vec_Flp for', len(Flp_lst)-1 ,'is', Vec_Flp  
	Flp_Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(Vec_Flp[0:len(FirAtt_lst)] == 1)].tolist()) &\
			   df2['Employer'].isin(SecAtt_lst[np.where(Vec_Flp[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)] == 1)].tolist())  &\
			   df2['Calendar Year'].isin(ThrAtt_lst[np.where(Vec_Flp[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)] == 1)].tolist())]
	Sal_list     = []
	ID_list      = []
	if (Flp_Ctx.shape[0] >= 20):
		for row in range(Flp_Ctx.shape[0]):
                    Sal_list.append(Flp_Ctx.iloc[row,7])
		    ID_list.append(Flp_Ctx.iloc[row,0])
                Score = math.exp(Epsilon *(pd.merge(Orgn_Ctx, Flp_Ctx, how='inner').shape[0]))
                Sal_arr= np.array(Sal_list)
                clf = LocalOutlierFactor(n_neighbors=20)
                Sal_outliers = clf.fit_predict(Sal_arr.reshape(-1,1))
		for outlier_finder in range(0, len(ID_list)):
                    if ((Sal_outliers[outlier_finder]==-1) and (ID_list[outlier_finder]==Queried_ID)):  
			Flp_lst.append([len(Flp_lst), Score, pd.merge(Orgn_Ctx, Flp_Ctx, how='inner').shape[0], np.zeros(len(Org_Vec))])
			for i in  range (len(Flp_lst[len(Flp_lst)-1][3])):    
				Flp_lst[len(Flp_lst)-1][3][i] = Vec_Flp[i]
			
       ###################################      Sampling form Exp Mech Result      #################################
elements = [elem[0] for elem in Flp_lst]
probabilities =[]
for prob in Flp_lst:
	probabilities.append(prob[1]/(sum ([prob[1] for prob in Flp_lst])))
ExpRes = np.random.choice(elements, 1, p = probabilities)  
Data_to_write.append(Flp_lst[ExpRes[0]][2]/max_ctx) 

t1 = time.time()
runtime = str(int((t1-t0) / 3600)) + ' hours and ' + str(int(((t1-t0) % 3600)/60)) + \
	' minutes and ' + str(((t1-t0) % 3600)%60) + ' seconds\n'
	    	   
writefinal(Data_to_write, str(int(sys.argv[1])), runtime, str(Queried_ID), max_ctx) 
print '\n\nThe required time for running the Random Walk algorithm is:', runtime
