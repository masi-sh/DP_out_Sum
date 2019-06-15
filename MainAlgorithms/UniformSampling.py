from __future__ import division
from mpmath import mp
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
Store_file = 'USampleDataPointsOutput.dat'

# Writing final data 
def writefinal(Data_to_write, randomness, runtime, ID, max_ctx):	
	ff = open(Store_file,'a+')
	fcntl.flock(ff, fcntl.LOCK_EX)
	np.savetxt(ff, np.column_stack(Data_to_write), fmt=('%7.5f'), header = 'UniSampling for query number: '+ randomness +\
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
max_ctx = Queries.iloc[Query_num]['Max']
print '\nmaximal context has the population :\n', max_ctx

        ############### The probability of adding an attribute value to the context  ###############
Flp_p         = 0.5
Flp_lst       = []
Data_to_write = []
###################################        Flip the context, 100 times            ###############################
Epsilon       = 0.1

t0 = time.time()
while len(Flp_lst)<100:
	print "len(Flp_lst) is: ", len(Flp_lst)
	Vec_Flp = np.zeros(len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst), dtype=np.int) 
	for Ctx_sprt in range (0, len(Vec_Flp)):
        	if (np.random.binomial(size=1, n=1, p= Flp_p)==1):
                	Vec_Flp[Ctx_sprt]=1
			
	Flp_Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(Vec_Flp[0:len(FirAtt_lst)] == 1)].tolist()) &\
		       df2['Employer'].isin(SecAtt_lst[np.where(Vec_Flp[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)] == 1)].tolist())  &\
		       df2['Calendar Year'].isin(ThrAtt_lst[np.where(Vec_Flp[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)] == 1)].tolist())]
	
	Sal_list     = []
	ID_list      = []
	if (Flp_Ctx.shape[0] >= 20):
		for row in range(Flp_Ctx.shape[0]):
                    Sal_list.append(Flp_Ctx.iloc[row,7])
		    ID_list.append(Flp_Ctx.iloc[row,0])
                Score = mp.exp(Epsilon *(Flp_Ctx.shape[0]))
                Sal_arr= np.array(Sal_list)
                clf = LocalOutlierFactor(n_neighbors=20)
                Sal_outliers = clf.fit_predict(Sal_arr.reshape(-1,1))
		for outlier_finder in range(0, len(ID_list)):
                    if ((Sal_outliers[outlier_finder]==-1) and (ID_list[outlier_finder]==Queried_ID)):  
			Flp_lst.append([len(Flp_lst), Score, Flp_Ctx.shape[0], np.zeros(len(Vec_Flp))])
                	for i in  range (len(Flp_lst[len(Flp_lst)-1][3])):      
				Flp_lst[len(Flp_lst)-1][3][i] = Vec_Flp[i]

       ###################################      Sampling form Exp Mech Result      #################################
elements = [elem[0] for elem in Flp_lst]	
probabilities =[]
for prob in Flp_lst:
	probabilities.append(prob[1]/(sum ([prob[1] for prob in Flp_lst])))

ExpRes = np.random.choice(elements, 1, p = probabilities)
print '\n\nThe number of candidates in Exponential mechanism range is:', len(Flp_lst)
print '\n\nIDs sampled from Exponential mechanism output are\n\n',  ExpRes

Data_to_write.append(Flp_lst[ExpRes[0]][2]/max_ctx) 

t1 = time.time()
runtime = str(int((t1-t0) / 3600)) + ' hours and ' + str(int(((t1-t0) % 3600)/60)) + \
	' minutes and ' + str(((t1-t0) % 3600)%60) + ' seconds\n'
	    	   
writefinal(Data_to_write, str(int(sys.argv[1])), runtime, str(Queried_ID), max_ctx) 
print '\n\nThe required time for running the Uniform Sampling algorithm is:', runtime

