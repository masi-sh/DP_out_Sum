from __future__ import division
import matplotlib
matplotlib.use('Agg')
import sys
#import gzip
import pandas as pd
import numpy as np
import cufflinks as cf
import plotly
import plotly.offline as py
import plotly.graph_objs as go
cf.go_offline()
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
import time
import fcntl
import random
import csv

Query_num = int(sys.argv[1])
# This file is filtered, no extra filtering required
df2 = pd.read_csv("~/DP_out_Sum/dataset/FilteredData.csv")
Query_file = '/home/sm2shafi/DP_out_Sum/MainAlgorithms/Queries.csv'
Queries = pd.read_csv(Query_file, 'rt', delimiter=',' , engine = 'python')
Store_file = 'BFSminDataPointsOutput.dat'

# Writing final data 
def writefinal(Data_to_write, randomness, runtime, ID):	
	ff = open(Store_file,'a+')
	fcntl.flock(ff, fcntl.LOCK_EX)
	np.savetxt(ff, np.column_stack(Data_to_write), fmt=('%7.5f'), header = randomness+ ' Generates outlier , ' + ID + ', BFSexp alg. takes' + runtime)
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

# Minimal Context, and transfer vector to use as intermediate variable in queue  
mnml_Vec = np.zeros(len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst))
mnml_Vec[np.where(FirAtt_lst == df2.values[Queried_ID][5])] = 1 
mnml_Vec[np.where(SecAtt_lst == df2.values[Queried_ID][4])[0]+len(FirAtt_lst)] = 1 
mnml_Vec[np.where(ThrAtt_lst == df2.values[Queried_ID][7])[0]+(len(FirAtt_lst)+len(SecAtt_lst))] = 1 
mnml_Ctx = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(mnml_Vec[0:len(FirAtt_lst)] == 1)].tolist()) &\
                   df2['Employer'].isin(SecAtt_lst[np.where(mnml_Vec[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)] == 1)].tolist())  &\
                   df2['Calendar Year'].isin(ThrAtt_lst[np.where(mnml_Vec[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)] == 1)].tolist())]

trsf_Vec = np.zeros(len(mnml_Vec))
trsf_Vec[np.where(FirAtt_lst == df2.values[Queried_ID][5])] = 1 
trsf_Vec[np.where(SecAtt_lst == df2.values[Queried_ID][4])[0]+len(FirAtt_lst)] = 1 
trsf_Vec[np.where(ThrAtt_lst == df2.values[Queried_ID][7])[0]+(len(FirAtt_lst)+len(SecAtt_lst))] = 1
# Initiating queue with Minimal Context informaiton 
Epsilon       = 0.001
effective_pop = 0
Min_Sal_list  = []
Min_ID_list   = []
for row in range(mnml_Ctx.shape[0]):
	Min_Sal_list.append(mnml_Ctx.iloc[row]['Salary Paid'])
	Min_ID_list.append(mnml_Ctx.iloc[row]['Unnamed: 0'])
			
Min_Sal_arr= np.array(Min_Sal_list)
clf = LocalOutlierFactor(n_neighbors=20)
Min_Sal_outliers = clf.fit_predict(Min_Sal_arr.reshape(-1,1))
for outlier_finder in range(0, len(Min_ID_list)):
	if ((Min_Sal_outliers[outlier_finder]==-1) and (Min_ID_list[outlier_finder]==Queried_ID)): 
		effective_pop = shape[0]
Min_Score = np.exp(Epsilon*(effective_pop))
Queue	= [[0, Min_Score, effective_pop, mnml_Vec]]
Data_to_write = [effective_pop/max_ctx]

###################################      Add to the minimal context, 100 times    ###############################
BFS_Flp = np.zeros(len(mnml_Vec)) 
t0      = time.time()

while len(Queue)<100: 	
	for i in  range (len(trsf_Vec)):      
		BFS_Flp[i] = trsf_Vec[i]
	
	effective_pop = 0
   	while any(np.array_equal(BFS_Flp[:],x[3][:]) for x in Queue):
        	Flp_bit  = random.randint(0,(len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)-1))
        	while  BFS_Flp[Flp_bit] == 1: 
            		Flp_bit  = random.randint(0,(len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)-1)) 
        	BFS_Flp[Flp_bit] = 1
		
	BFS_Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(BFS_Flp[0:len(FirAtt_lst)] == 1)].tolist()) &\
			   df2['Employer'].isin(SecAtt_lst[np.where(BFS_Flp[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)] == 1)].tolist())  &\
			   df2['Calendar Year'].isin(ThrAtt_lst[np.where(BFS_Flp[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)] == 1)].tolist())]
	Sal_list     = []
	ID_list      = []
	if (BFS_Ctx.shape[0] >= 20):
		for row in range(BFS_Ctx.shape[0]):
                    Sal_list.append(BFS_Ctx.iloc[row]['Salary Paid'])
		    ID_list.append(BFS_Ctx.iloc[row]['Unnamed: 0'])
			
                Sal_arr= np.array(Sal_list)
                clf = LocalOutlierFactor(n_neighbors=20)
                Sal_outliers = clf.fit_predict(Sal_arr.reshape(-1,1))
		for outlier_finder in range(0, len(ID_list)):
                    if ((Sal_outliers[outlier_finder]==-1) and (ID_list[outlier_finder]==Queried_ID)): 
			effective_pop = shape[0]
	        Score = np.exp(Epsilon*(effective_pop))
		Queue.append([len(Queue), Score, effective_pop, np.zeros(len(mnml_Vec)])
		for i in  range (len(Queue[len(Queue)-1][3])):      
			Queue[len(Queue)-1][3][i] = BFS_Flp[i]
		print '\n Len(Queue) is = ', len(Queue), '\n The private context candidates are: \n', Queue
	###################################       Sampling form the Queue ###############################
	elements = [elem[0] for elem in Queue]	
	probabilities = [prob[1] for prob in Queue]/(sum ([prob[1] for prob in Queue]))
	ExpRes = np.random.choice(elements, 1, p = probabilities)
	ExpRes = np.random.choice(elements, 1, p = probabilities)
    	for child in range(0, len(Queue)):
        	if Queue[child][0] == ExpRes[0]:
            		Q_indx = child          
	for i in  range (len(Queue[Q_indx][3])):            
		trsf_Vec[i]  = Queue[Q_indx][3][i]
	
	print 'The candidate picked form the Q is ', ExpRes[0], 'th, with context ', Queue[Q_indx][3][:],\
	' and has ', Queue[Q_indx][2], 'population'
			      
	Data_to_write.append(Queue[Q_indx][2]/max_ctx)
	
t1 = time.time()

runtime = str(int((t1-t0) / 3600)) + ' hours and ' + str(int(((t1-t0) % 3600)/60)) + \
	' minutes and ' + str(((t1-t0) % 3600)%60) + ' seconds\n'   
writefinal(Data_to_write, str(int(sys.argv[1])), runtime, str(Queried_ID))	

print '\n The final Queue is \n', Queue     
print '\n The BFS runtime, starting from minimal context and choosing randomly among childern in each layer is \n', runtime
