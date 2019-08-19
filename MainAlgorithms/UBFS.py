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
df2 = pd.read_csv("~/DP_out_Sum/dataset/FilteredData.csv")
Query_file = '/home/sm2shafi/DP_out_Sum/MainAlgorithms/Queries.csv'
Queries = pd.read_csv(Query_file, 'rt', delimiter=',' , engine = 'python')
Store_file = 'UBFS.dat'

# Finds the maximal context for the Queried_ID      
def maxctx(Ref_file, Queried_ID):
	print '\nChecking for the maximal context ... \n'
	max = 0
	out_size = 0
	#line_num = 0
	size = 0
	#Ctx_line = 0
	with open(Ref_file,'rt') as f:
        	for num, line in enumerate(f, 1):
                	if line.split(' ')[0].strip()=="Matching":
				#Ctx_line = num
                        	size = int((line.split(' '))[5].strip(':\n'))
			elif line.strip().startswith("ID"):
				if line.split(' ')[3].strip('#')==str(Queried_ID):
					out_size = size
					#Valid_line = Ctx_line
                	if (max < out_size):
				max = out_size
				#line_num = Valid_line 
				print "\nmax so far is :", max, "   at time: ", time.time()
	f.close()
	return max;

# Writing final data 
def writefinal(Data_to_write, randomness, runtime, ID):	
	ff = open(Store_file,'a+')
	fcntl.flock(ff, fcntl.LOCK_EX)
	np.savetxt(ff, np.column_stack(Data_to_write), fmt=('%7.5f'), header = randomness+ ' Generates outlier , ' + ID + ', \
	UBFS alg. takes' + runtime)
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
#max_ctx = Queries.iloc[Query_num]['Max']
#print '\nmaximal context has the population :\n', max_ctx

# Making Queue of samples and initiating it, with Org_Vec   
Org_Vec       = np.zeros(len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst))

# polishing Ctx in Query_file and reading Org_Vec from it
Queries['Ctx'] = Queries['Ctx'].replace({'\n': ''}, regex=True)
Org_Str = Queries.iloc[Query_num]['Ctx'][1:-2].strip('[]').replace('.','').replace(' ', '')
for i in range(len(Org_Vec)):
	if (Org_Str[i] =='1'):
		Org_Vec[i] = 1
		
Orgn_Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(Org_Vec[0:len(FirAtt_lst)] == 1)].tolist()) &\
		    df2['Employer'].isin(SecAtt_lst[np.where(Org_Vec[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)] == 1)].tolist())  &\
                    df2['Calendar Year'].isin(ThrAtt_lst[np.where(Org_Vec[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)] == 1)].tolist())]
# Utility here is intersection with original context's size
max_ctx = Orgn_Ctx.shape[0]
# Initiating queue with Org_ctx informaiton 
Epsilon       = 0.001
Queue	      = [[0, mp.exp(Epsilon *(Orgn_Ctx.shape[0])), Orgn_Ctx.shape[0], Org_Vec]]
# Samples start with org_vec info
Data_to_write = [(Queue[0][2])/max_ctx]

# Running the BFS_Alg to form a queue of 100 elements
t0 = time.time()
def BFS_Alg(Org_Vec, Queue, Data_to_write, Epsilon, max_ctx):
	Stats = np.full(2**len(Org_Vec)+1, True, dtype=bool)
	BFS_Vec      = np.zeros(len(Org_Vec))
	for i in range(len(Org_Vec)):
		BFS_Vec[i]  = Org_Vec[i]
	Stack = [[0, mp.exp(Epsilon *(Orgn_Ctx.shape[0])), Orgn_Ctx.shape[0], Org_Vec]]
	BFS_Flp = np.zeros(len(Org_Vec))
	#Q_indx        = 0
	#index         = 0
	termination_threshold = 500
	Terminator    = 0
	while len(Queue)<50:
		New_Ctx = int(str(BFS_Vec).replace(',', '').replace(' ','').replace('.','').replace('\n','')[1:-1],2)
		Stats[New_Ctx] = False 
		print 'len(Queue) is', len(Queue)
    		Terminator += 1
    		if (Terminator>termination_threshold):
			break
    		Addtosamples    = False
		sub_q    = []
		for Flp_bit in range(0,(len(BFS_Vec))):
			Sub_Sal_list = []
			Sub_ID_list  = []
			for i in  range (len(BFS_Vec)):
				BFS_Flp[i] = BFS_Vec[i]
			BFS_Flp[Flp_bit] = 1 - BFS_Flp[Flp_bit]
			BFS_Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(BFS_Flp[0:len(FirAtt_lst)] == 1)].tolist()) &\
					   df2['Employer'].isin(SecAtt_lst[np.where(BFS_Flp[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)] == 1)].tolist())  &\
					   df2['Calendar Year'].isin(ThrAtt_lst[np.where(BFS_Flp[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)] == 1)].tolist())]
			if (Stats[int(str(BFS_Flp).replace(',', '').replace(' ','').replace('.','').replace('\n','')[1:-1],2)] == True and (BFS_Ctx.shape[0] > 20)):
				for row in range(BFS_Ctx.shape[0]):
					Sub_Sal_list.append(BFS_Ctx.iloc[row,7])
					Sub_ID_list.append(BFS_Ctx.iloc[row,0])		
				Sub_Sal_arr= np.array(Sub_Sal_list)
				clf = LocalOutlierFactor(n_neighbors=20)
				Sub_Sal_outliers = clf.fit_predict(Sub_Sal_arr.reshape(-1,1))
				for outlier_finder in range(0, len(Sub_ID_list)):
					if ((Sub_Sal_outliers[outlier_finder]==-1) and (Sub_ID_list[outlier_finder]==Queried_ID)):
						Sub_Score = mp.exp(Epsilon*(pd.merge(Orgn_Ctx, BFS_Ctx, how='inner').shape[0]))
          					sub_q.append([Flp_bit ,Sub_Score , pd.merge(Orgn_Ctx, BFS_Ctx, how='inner').shape[0], np.zeros(len(Org_Vec))])
						for i in  range (len(sub_q[len(sub_q)-1][3])):      
							sub_q[len(sub_q)-1][3][i] = BFS_Flp[i]	
						Stats[int(str(BFS_Flp).replace(',', '').replace(' ','').replace('.','').replace('\n','')[1:-1],2)] = False 
		# Sampling from sub_queue(sampling in each layer) 
		for i in range (len(Stack)):
			if (Stack[i][3].tolist() == BFS_Vec.tolist()):
				Stack.pop(i)
				break
		if (len(sub_q)>1):
			for i in range (len(sub_q)):
				Stack.append(sub_q[i])
		Sub_elements = [elem for elem in range(len(Stack))]
		Sub_probabilities =[]
    		for prob in Stack:
			Sub_probabilities.append(prob[1]/(sum ([prob[1] for prob in Stack])))
		SubRes = np.random.choice(Sub_elements, 1, p = Sub_probabilities)
		Queue.append([len(Queue), Stack[SubRes[0]][1], Stack[SubRes[0]][2], Stack[SubRes[0]][3][:]])
		for i in range(len(Stack[SubRes[0]][3])):
			BFS_Vec[i] = Stack[SubRes[0]][3][i]
		Addtosamples = True
		Terminator = 0
		
		print '\n len(Queue) is = ',len(Queue), '\n The private context candidates are: \n', Queue
		# Continuing form the Queue
		print 'The candidate picked form the Q is ', Queue[len(Queue)-1][0], 'th, with context ', Queue[len(Queue)-1][3][:],\
		' and has ', Queue[len(Queue)-1][2], 'population'
		if (Addtosamples):
			Data_to_write.append(Queue[len(Queue)-1][2]/max_ctx) 
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