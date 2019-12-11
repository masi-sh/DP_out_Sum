from __future__ import division
import sys
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
from mpmath import mp

Query_num = int(sys.argv[1])
# This file is filtered, no extra filtering required
df2 = pd.read_csv("~/DP_out_Sum/dataset/MurderData.csv")
Query_file = '/home/sm2shafi/DP_out_Sum/Murder/LOF/MLQueries.csv'
Queries = pd.read_csv(Query_file, 'rt', delimiter=',' , engine = 'python')
Store_file = 'MLDFS-e4.dat'

# Writing final data 
def writefinal(Data_to_write, randomness, runtime, ID, max_ctx):	
	ff = open(Store_file,'a+')
	fcntl.flock(ff, fcntl.LOCK_EX)
	np.savetxt(ff, np.column_stack(Data_to_write), fmt=('%7.5f'), header = 'DFS for query number: '+ randomness +\
	'for outlier' + ID + 'with Ctx_max '+ str(max_ctx) + 'on Murder dataset takes ' + runtime)	
	fcntl.flock(ff, fcntl.LOCK_UN)
	ff.close()
	return;

FirAtt_lst = df2['Weapon'].unique()
SecAtt_lst = df2['State'].unique()
ThrAtt_lst = df2['AgencyType'].unique()
	
# Reading a Queried_ID from the list in the Queries file
Queried_ID = Queries.iloc[Query_num]['Outlier']
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
Orgn_Ctx  = df2.loc[df2['Weapon'].isin(FirAtt_lst[np.where(Org_Vec[0:len(FirAtt_lst)] == 1)].tolist()) &\
                    df2['State'].isin(SecAtt_lst[np.where(Org_Vec[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)] == 1)].tolist())  &\
                    df2['AgencyType'].isin(ThrAtt_lst[np.where(Org_Vec[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)] == 1)].tolist())]
# Making Queue of samples and initiating it, with Org_Vec
# Initiating queue with Org_ctx informaiton
Epsilon       = 0.004
Queue	      = [[0, mp.exp(Epsilon *(Orgn_Ctx.shape[0])), Orgn_Ctx.shape[0], Org_Vec]]
# Samples start with org_vec info
Data_to_write = []
Stack = [[0, mp.exp(Epsilon *(Orgn_Ctx.shape[0])), Orgn_Ctx.shape[0], Org_Vec]]
# Make the queue by DFS traverse from ctx_org by exp through children, 100 times 
t0       = time.time()
def DFS_Alg(Org_Vec, Queue, Data_to_write, Epsilon, max_ctx):
	Visited = []
	contexts = [Org_Vec]
	termination_threshold =500
	Terminator = 0
	while len(Visited)<100:
		Terminator += 1
   		if (Terminator>termination_threshold):
			break
		BFS_Vec      = np.zeros(len(Org_Vec))
		for i in range(len(Org_Vec)):
			BFS_Vec[i]  = Stack[len(Stack)-1][3][i]
		Visited.append(np.zeros(len(Org_Vec)))
		for i in range(len(Org_Vec)):
			Visited[len(Visited)-1][i]  = Stack[len(Stack)-1][3][i]	
		Queue.append(Stack[len(Stack)-1])
		BFS_Flp  = np.zeros(len(Org_Vec)) 
		sub_q    = []
		for Flp_bit in range(0,(len(Org_Vec))):
			Sub_Sal_list = []
			Sub_ID_list  = []
			for i in  range (len(BFS_Vec)):      
				BFS_Flp[i] = BFS_Vec[i]
			BFS_Flp[Flp_bit] = 1 - BFS_Flp[Flp_bit]
			BFS_Ctx  = df2.loc[df2['Weapon'].isin(FirAtt_lst[np.where(BFS_Flp[0:len(FirAtt_lst)] == 1)].tolist()) &\
					   df2['State'].isin(SecAtt_lst[np.where(BFS_Flp[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)] == 1)].tolist())  &\
					   df2['AgencyType'].isin(ThrAtt_lst[np.where(BFS_Flp[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)] == 1)].tolist())]
			if ((not any(np.array_equal(BFS_Flp[:],x[:]) for x in Visited)) and (not any(np.array_equal(BFS_Flp[:],x[:]) for x in contexts)) and (BFS_Ctx.shape[0] > 20)):	
				for row in range(BFS_Ctx.shape[0]):
					Sub_Sal_list.append(BFS_Ctx.iloc[row,4])
					Sub_ID_list.append(BFS_Ctx.iloc[row,0])		
				Sub_Sal_arr= np.array(Sub_Sal_list)
				clf = LocalOutlierFactor(n_neighbors=20)
				Sub_Sal_outliers = clf.fit_predict(Sub_Sal_arr.reshape(-1,1))
				for outlier_finder in range(0, len(Sub_ID_list)):
					if ((Sub_Sal_outliers[outlier_finder]==-1) and (Sub_ID_list[outlier_finder]==Queried_ID)):
						Sub_Score = mp.exp(Epsilon *(BFS_Ctx.shape[0]))
          					sub_q.append([Flp_bit ,Sub_Score , BFS_Ctx.shape[0], np.zeros(len(Org_Vec))])
						for i in  range (len(sub_q[len(sub_q)-1][3])):      
							sub_q[len(sub_q)-1][3][i] = BFS_Flp[i]				
		# Sampling from sub_queue(sampling in each layer) 
		if not sub_q:
			Stack.remove(Stack[len(Stack)-1])
		else:       
			Sub_elements = [elem[0] for elem in sub_q]
			Sub_probabilities =[]
    			for prob in sub_q:
				Sub_probabilities.append(prob[1]/(sum ([prob[1] for prob in sub_q])))
			SubRes = np.random.choice(Sub_elements, 1, p = Sub_probabilities)
			for child in range(0, len(sub_q)):
				if sub_q[child][0] == SubRes[0]:
					Q_indx = child	
			Stack.append([len(Stack), sub_q[Q_indx][1],sub_q[Q_indx][2] ,np.zeros(len(BFS_Vec))])
			for i in  range (len(BFS_Vec)):      
				Stack[len(Stack)-1][3][i] = sub_q[Q_indx][3][i]	
			contexts.append(np.zeros(len(Org_Vec)))
			for i in range(len(Org_Vec)):
				contexts[len(contexts)-1][i]  = sub_q[Q_indx][3][i]	
				
	# Exp mechanism on the visited nodes
	for i in  range (len(Queue)):   
		Queue[i][0] = i
	elements = [elem for elem in range(len(Queue))]
	probabilities =[]
	for prob in Queue:
		probabilities.append(prob[1]/(sum ([prob[1] for prob in Queue])))
	Res = np.random.choice(elements, 1, p = probabilities)
	Data_to_write.append(Queue[Res[0]][2]/max_ctx)			
	return;

DFS_Alg(Org_Vec, Queue, Data_to_write, Epsilon, max_ctx)
t1 = time.time()
runtime = str(int((t1-t0) / 3600)) + ' hours and ' + str(int(((t1-t0) % 3600)/60)) + \
' minutes and ' + str(((t1-t0) % 3600)%60) + ' seconds\n'	    	   
writefinal(Data_to_write, str(Query_num), runtime, str(Queried_ID), max_ctx)	
