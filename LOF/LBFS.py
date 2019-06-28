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
# This file is filtered, no extra filtering required
df2 = pd.read_csv("home/sm2shafi/DP_out_Sum/Grubbs/ToyData.csv")
Query_file = '/home/sm2shafi/DP_out_Sum/LOF/TLQueries.csv'
Queries = pd.read_csv(Query_file, 'rt', delimiter=',' , engine = 'python')
Store_file = 'LBFS.dat'

# Writing final data 
def writefinal(Data_to_write, randomness, runtime, ID):	
	ff = open(Store_file,'a+')
	fcntl.flock(ff, fcntl.LOCK_EX)
	np.savetxt(ff, np.column_stack(Data_to_write), fmt=('%7.5f'), header = randomness+ ' Generates outlier , ' + ID + ', \
	BFS alg. takes' + runtime)
	fcntl.flock(ff, fcntl.LOCK_UN)
	ff.close()
	return;

### Data is filtered, no more polishing required

FirAtt_lst = df2['Job Title'].unique()
SecAtt_lst = df2['Employer'].unique()
ThrAtt_lst = df2['Calendar Year'].unique()
	
Queried_ID = Queries.iloc[Query_num]['Outlier']
print '\n\n Outlier\'s ID in the original context is: ', Queried_ID
# finding maximal context's size for queried_ID
max_ctx = Queries.iloc[Query_num]['Max']
print '\nmaximal context has the population :\n', max_ctx

FirAtt_lst = df2['Job Title'].unique()
SecAtt_lst = df2['Employer'].unique()
ThrAtt_lst = df2['Calendar Year'].unique()
# Supersets for each attribute
FirAtt_Sprset = sum(map(lambda r: list(combinations(FirAtt_lst[0:], r)), range(1, len(FirAtt_lst[0:])+1)), [])
SecAtt_Sprset = sum(map(lambda r: list(combinations(SecAtt_lst[0:], r)), range(1, len(SecAtt_lst[0:])+1)), [])
ThrAtt_Sprset = sum(map(lambda r: list(combinations(ThrAtt_lst[0:], r)), range(1, len(ThrAtt_lst[0:])+1)), [])	
# Reading and polishing Ctx in Query_file
context  = int(Queries.iloc[Query_num]['Ctx'])
iii = context%1000
jjj = (context//1000)%1000
zzz = context//1000000
Orgn_Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_Sprset[iii]) & df2['Employer'].isin(SecAtt_Sprset[jjj]) &\
		    df2['Calendar Year'].isin(ThrAtt_Sprset[zzz])]

# Making Queue of samples and initiating it, with Org_Vec   
Org_Vec  = np.zeros(len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst))
temp_Vec = np.zeros(len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst))

Org_Vec[np.where(np.isin(FirAtt_lst[0:len(FirAtt_lst)], FirAtt_Sprset[iii]))] = 1
temp_Vec[np.where(np.isin(SecAtt_lst[0:len(SecAtt_lst)], SecAtt_Sprset[jjj]))] = 1
Org_Vec[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)] = temp_Vec[0:len(SecAtt_lst)]
temp_Vec = np.zeros(len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst))
temp_Vec[np.where(np.isin(ThrAtt_lst[0:len(ThrAtt_lst)], ThrAtt_Sprset[zzz]))] = 1
Org_Vec[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)] = temp_Vec[0:len(ThrAtt_lst)]

# Initiating queue with Org_ctx informaiton 
Epsilon       = 0.001
Queue	      = [[0, mp.exp(Epsilon *(Orgn_Ctx.shape[0])), Orgn_Ctx.shape[0], Org_Vec]]
# Samples start with org_vec info
Data_to_write = [(Queue[0][2])/max_ctx]

# Running the BFS_Alg to form a queue of 100 elements
t0 = time.time()
def BFS_Alg(Org_Vec, Queue, Data_to_write, Epsilon, max_ctx):
	BFS_Flp       = np.zeros(len(Org_Vec)) 
	Q_indx        = 0
	index         = 0
	termination_threshold = 500
	Terminator    = 0
	while len(Queue)<100:
    		Terminator += 1
    		if (Terminator>termination_threshold):
			break
    		Addtosamples    = False
   		print '\nSampling & Queueing...  \n',
    		for i in  range (len(Queue[Q_indx][3])):      
        		BFS_Flp[i]        = Queue[Q_indx][3][i]
    		while any(np.array_equal(BFS_Flp[:],x[3][:]) for x in Queue):
			print '\nThis is already on the queue too!'
        		for i in  range (len(Queue[Q_indx][3])):      
            			BFS_Flp[i]    = Queue[Q_indx][3][i]
        		Flp_bit           = random.randint(0,(len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)-1))
        		BFS_Flp[Flp_bit]  = 1 - BFS_Flp[Flp_bit]
    		BFS_Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(BFS_Flp[0:len(FirAtt_lst)] == 1)].tolist()) &\
				   df2['Employer'].isin(SecAtt_lst[np.where(BFS_Flp[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)] == 1)].tolist())  &\
				   df2['Calendar Year'].isin(ThrAtt_lst[np.where(BFS_Flp[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)] == 1)].tolist())]
    		Sal_list     = []
    		ID_list      = []
    		if (BFS_Ctx.shape[0] >= 20):
        		for row in range(BFS_Ctx.shape[0]):
            			Sal_list.append(BFS_Ctx.iloc[row,7])
            			ID_list.append(BFS_Ctx.iloc[row,0])

        		Sal_arr= np.array(Sal_list)
        		clf = LocalOutlierFactor(n_neighbors=20)
        		Sal_outliers = clf.fit_predict(Sal_arr.reshape(-1,1))
        		for outlier_finder in range(0, len(ID_list)):
            			if ((Sal_outliers[outlier_finder]==-1) and (ID_list[outlier_finder]==Queried_ID)): 
                			Score = mp.exp(Epsilon *(BFS_Ctx.shape[0]))
                			Queue.append([len(Queue), Score, BFS_Ctx.shape[0], np.zeros(len(Org_Vec))])
					Addtosamples = True
					Terminator   = 0
                			for i in  range (len(Queue[len(Queue)-1][3])):      
                    				Queue[len(Queue)-1][3][i]  = BFS_Flp[i]

   		# Sampling form the Queue
    		elements = [elem[0] for elem in Queue]
    		probabilities =[]
    		for prob in Queue:
			probabilities.append(prob[1]/(sum ([prob[1] for prob in Queue])))
    		ExpRes = np.random.choice(elements, 1, p = probabilities)
    		for child in range(0, len(Queue)):
        		if Queue[child][0] == ExpRes[0]:
            			Q_indx = child
    		if (Addtosamples):
			Data_to_write.append((Queue[Q_indx][2])/max_ctx) 
			print 'In BFS_Alg, Data_to_write is: ', Data_to_write
	return;

BFS_Alg(Org_Vec, Queue, Data_to_write, Epsilon, max_ctx)
print 'Out BFS_Alg, Data_to_write is: ', Data_to_write

Data_to_write = np.append(Data_to_write , np.zeros(100 - len(Data_to_write)))
#print 'The candidate picked form the Q is ', ExpRes[0], 'th, with context ', Queue[ExpRes[0]][3][:],' and has ', Queue[ExpRes[0]][2], 'population'
t1 = time.time()
runtime = str(int((t1-t0) / 3600)) + ' hours and ' + str(int(((t1-t0) % 3600)/60)) + \
	' minutes and ' + str(((t1-t0) % 3600)%60) + ' seconds\n'
	    	   
writefinal(Data_to_write, str(Query_num), runtime, str(Queried_ID))	
#print '\n The final Queue is \n', Queue     
print '\n The BFS runtime, starting from org_ctx and choosing randomly one among childern in each layer is \n', runtime