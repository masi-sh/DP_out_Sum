from __future__ import division
import matplotlib
matplotlib.use('Agg')
import sys
#import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
Store_file = 'BFSexpDataPointsOutput.dat'

# Writing final data 
def writefinal(Data_to_write, randomness, runtime, ID, max_ctx):	
	ff = open(Store_file,'a+')
	fcntl.flock(ff, fcntl.LOCK_EX)
	np.savetxt(ff, np.column_stack(Data_to_write), fmt=('%7.5f'), header = 'BFSexp for query number: '+ randomness +\
	'for outlier' + ID + 'with Ctx_max '+ str(max_ctx) + 'takes ' + runtime)	
	fcntl.flock(ff, fcntl.LOCK_UN)
	ff.close()
	return;

FirAtt_lst = df2['Job Title'].unique()
SecAtt_lst = df2['Employer'].unique()
ThrAtt_lst = df2['Calendar Year'].unique()
	
# Reading a Queried_ID from the list in the Queries file
Queried_ID = Queries.[Query_num]['Outlier']
print '\n\n Outlier\'s ID in the original context is: ', Queried_ID
# finding maximal context's size for queried_ID
max_ctx = Queries.iloc[Query_num]['Max']
print '\nmaximal context has the population :\n', max_ctx

# Making Queue of samples and initiating it, with Org_Vec  
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

# Making Queue of samples and initiating it, with Org_Vec
# BFS_Vec is the transferring vector 
# Initiating queue with Org_ctx informaiton
Epsilon       = 0.001
Queue	      = [[0, math.exp(Epsilon *(Orgn_Ctx.shape[0])), Orgn_Ctx.shape[0], Org_Vec]]
# Samples start with org_vec info
Data_to_write = [(Queue[0][2])/max_ctx]

BFS_Vec      = np.zeros(len(Org_Vec))
for i in range(len(Org_Vec)):
	BFS_Vec[i]  = Org_Vec[i]

#  Make the queue by BFS traverse from ctx_org by exp through children, 100 times 
t0       = time.time()
BFS_Flp  = np.zeros(len(Org_Vec)) 
termination_threshold =500
Terminator = 0
while len(Queue)<100:  
	Terminator += 1
   	if (Terminator>termination_threshold):
		break
	Addtosamples = False
	sub_q    = []
	for Flp_bit in range(0,(len(Org_Vec))):
		Sub_Sal_list = []
		Sub_ID_list  = []
		for i in  range (len(BFS_Vec)):      
			BFS_Flp[i] = BFS_Vec[i]
		#if BFS_Flp[Flp_bit] == 0:
		BFS_Flp[Flp_bit] = 1 - BFS_Flp[Flp_bit]
		BFS_Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(BFS_Flp[0:len(FirAtt_lst)] == 1)].tolist()) &\
				   df2['Employer'].isin(SecAtt_lst[np.where(BFS_Flp[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)] == 1)].tolist())  &\
				   df2['Calendar Year'].isin(ThrAtt_lst[np.where(BFS_Flp[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)] == 1)].tolist())]
		if (BFS_Ctx.shape[0] > 20):
			for row in range(BFS_Ctx.shape[0]):
				Sub_Sal_list.append(BFS_Ctx.iloc[row, 7])
				Sub_ID_list.append(BFS_Ctx.iloc[row, 0])		
			Sub_Sal_arr= np.array(Sub_Sal_list)
			clf = LocalOutlierFactor(n_neighbors=20)
			Sub_Sal_outliers = clf.fit_predict(Sub_Sal_arr.reshape(-1,1))
			for outlier_finder in range(0, len(Sub_ID_list)):
				if ((Sub_Sal_outliers[outlier_finder]==-1) and (Sub_ID_list[outlier_finder]==Queried_ID)):
					Sub_Score = math.exp(Epsilon *(BFS_Ctx.shape[0]))
          				sub_q.append([Flp_bit ,Sub_Score , BFS_Ctx.shape[0], np.zeros(len(Org_Vec))])
					for i in  range (len(sub_q[len(sub_q)-1][3])):      
						sub_q[len(sub_q)-1][3][i] = BFS_Flp[i]
			
	#######################       Sampling from sub_queue(sampling in each layer)        ##################################
	Sub_elements = [elem[0] for elem in sub_q]
	Sub_probabilities =[]
    	for prob in sub_q:
		Sub_probabilities.append(prob[1]/(sum ([prob[1] for prob in sub_q])))
	SubRes = np.random.choice(Sub_elements, 1, p = Sub_probabilities)
	for child in range(0, len(sub_q)):
		if sub_q[child][0] == SubRes[0]:
			Q_indx = child
	while not any(np.array_equal(sub_q[Q_indx][3][:],x[3]) for x in Queue):
		Queue.append([len(Queue), sub_q[Q_indx][1], sub_q[Q_indx][2], sub_q[Q_indx][3][:]])
		Addtosamples = True
		Terminator = 0

	print '\n len(Queue) is = ',len(Queue), '\n The private context candidates are: \n', Queue
	##################################       Sampling form the Queue ###############################
	elements = [elem[0] for elem in Queue]	
	probabilities =[]
    	for prob in Queue:
		probabilities.append(prob[1]/(sum ([prob[1] for prob in Queue])))
	ExpRes = np.random.choice(elements, 1, p = probabilities)
	for child in range(0, len(Queue)):
      		if Queue[child][0] == ExpRes[0]:
        		QQ_indx = child
	for i in  range (len(Queue[QQ_indx][3])): 
		BFS_Vec[i]  = Queue[QQ_indx][3][i]
	print 'The candidate picked form the Q is ', ExpRes[0], 'th, with context ', Queue[QQ_indx][3][:],\
	' and has ', Queue[QQ_indx][2], 'population'
	if (Addtosamples):
		Data_to_write.append(Queue[QQ_indx][2]/max_ctx) 
	
	###################################       Writing final data ###############################
Data_to_write = np.append(Data_to_write , np.zeros(100 - len(Data_to_write)))
t1 = time.time()
runtime = str(int((t1-t0) / 3600)) + ' hours and ' + str(int(((t1-t0) % 3600)/60)) + \
' minutes and ' + str(((t1-t0) % 3600)%60) + ' seconds\n'
	    	   
writefinal(Data_to_write, str(int(sys.argv[1])), runtime, str(Queried_ID), max_ctx)	
