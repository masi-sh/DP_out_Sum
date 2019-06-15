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
import hashlib

Query_num = int(sys.argv[1])
random.seed(50*Query_num)
# This file is filtered, no extra filtering required
df2 = pd.read_csv("~/DP_out_Sum/dataset/FilteredData.csv")
Query_file = '/home/sm2shafi/DP_out_Sum/dataset/RndQueries.csv'
Queries = pd.read_csv(Query_file, 'rt', delimiter=',' , engine = 'python')
Store_file = 'RndBFS.dat'

def hash_calc(i, j, z, ID):
        hash_value = hashlib.md5(str(i+1000*j+1000000*z)+str(ID))
        hash_hex = hash_value.hexdigest()
        #:as_int = int(hash_hex[30:32],16)
        #return (as_int%128==0);
	return (hash_hex[30:32] == '80' or hash_hex[30:32] == '00');

# Finding the index of an attribute list in Att_Sprset 
def Find_index(Att_Sprset, Att_Flp):
	index = 100000
	for x in range(len(Att_Sprset)):
		if np.array_equal(Att_Sprset[x],Att_Flp):
			index =x
			break
	return index;

# Writing final data 
def writefinal(Data_to_write, randomness, runtime, ID):	
	ff = open(Store_file,'a+')
	fcntl.flock(ff, fcntl.LOCK_EX)
	np.savetxt(ff, np.column_stack(Data_to_write), fmt=('%7.5f'), header = randomness+ ' Generates outlier , ' + ID + ', \
	RndBFS alg. takes' + runtime)
	fcntl.flock(ff, fcntl.LOCK_UN)
	ff.close()
	return;

# Reading queried_ID and its maximal context's size
Queried_ID = Queries.iloc[Query_num]['Outlier']
print '\n\n Outlier\'s ID in the original context is: ', Queried_ID
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
Queue	      = [[0, math.exp(Epsilon *(Orgn_Ctx.shape[0])), Orgn_Ctx.shape[0], Org_Vec]]
# Samples start with org_vec info
Data_to_write = [(Queue[0][2])/max_ctx]

# Running the BFS_Alg to form a queue of 100 elements
t0 = time.time()
def BFS_Alg(Org_Vec, Queue, Data_to_write, Epsilon, max_ctx):
	FirAtt_Flp = []
	SecAtt_Flp = []
	ThrAtt_Flp = []
	BFS_Flp    = np.zeros(len(Org_Vec)) 
	Q_indx     = 0
	index      = 0
	termination_threshold = 5000
	Terminator  = 0
	while len(Queue)<100:
    		Terminator += 1
		print 'Terminator', Terminator
    		if (Terminator>termination_threshold):
			break
    		Addtosamples    = False
   		print '\nSampling & Queueing...  \n',
    		for i in  range (len(Queue[Q_indx][3])):      
        		BFS_Flp[i]        = Queue[Q_indx][3][i]
    		while any(np.array_equal(BFS_Flp[:],x[3][:]) for x in Queue):
			print '\nThis is already on the queue too!'
        		for i in  range (len(Queue[Q_indx][3])):      
            			BFS_Flp[i] = Queue[Q_indx][3][i]
        		Flp_bit = random.randint(0,(len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)-1))
        		BFS_Flp[Flp_bit]  = 1 - BFS_Flp[Flp_bit]
			
		FirAtt_Flp = FirAtt_lst[np.where(BFS_Flp[0:len(FirAtt_lst)] == 1)]
		SecAtt_Flp = SecAtt_lst[np.where(BFS_Flp[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)] == 1)]	
		ThrAtt_Flp = ThrAtt_lst[np.where(BFS_Flp[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)] == 1)]
		
		iii = Find_index(FirAtt_Sprset, FirAtt_Flp)
		jjj = Find_index(SecAtt_Sprset, SecAtt_Flp)
		zzz = Find_index(ThrAtt_Sprset, ThrAtt_Flp)
		
    		BFS_Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_Sprset[iii]) & df2['Employer'].isin(SecAtt_Sprset[jjj]) &\
				   df2['Calendar Year'].isin(ThrAtt_Sprset[zzz])]
    		ID_list  = []
    		if (BFS_Ctx.shape[0] >= 20):
			for row in range(BFS_Ctx.shape[0]):
                        	if hash_calc(iii, jjj, zzz, Queried_ID):
					Score = math.exp(Epsilon *(BFS_Ctx.shape[0]))
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
			print 'Out RndBFS_Alg, Data_to_write is: ', Data_to_write
	return;

BFS_Alg(Org_Vec, Queue, Data_to_write, Epsilon, max_ctx)
print 'Out RndBFS_Alg, Data_to_write is: ', Data_to_write

Data_to_write = np.append(Data_to_write , np.zeros(100 - len(Data_to_write)))
#print 'The candidate picked form the Q is ', ExpRes[0], 'th, with context ', Queue[ExpRes[0]][3][:],' and has ', Queue[ExpRes[0]][2], 'population'
t1 = time.time()
runtime = str(int((t1-t0) / 3600)) + ' hours and ' + str(int(((t1-t0) % 3600)/60)) + \
	' minutes and ' + str(((t1-t0) % 3600)%60) + ' seconds\n'
	    	   
writefinal(Data_to_write, str(Query_num), runtime, str(Queried_ID))	
#print '\n The final Queue is \n', Queue     
print '\n The BFS runtime, starting from org_ctx and choosing randomly one among childern in each layer is \n', runtime
