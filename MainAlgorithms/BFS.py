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
#outputname  = 'Outputs/output'+sys.argv[1]+'.txt'
#Maxfilename = 'Max.txt'

# To get the same original contexts in all files
random.seed(100*int(sys.argv[1]))
# This file is filtered, no extra filtering required
df2 = pd.read_csv("~/DP_out_Sum/dataset/FilteredData.csv")
Ref_file = '/home/sm2shafi/Reffile.txt'
Store_file = 'BFSDataPointsOutput.dat'

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
	np.savetxt(ff, np.column_stack(Data_to_write), fmt=('%7.1f'), header = randomness+ ' Generates outlier , ' + ID + ', BFSexp alg. takes' + runtime)
	fcntl.flock(ff, fcntl.LOCK_UN)
	ff.close()
	return;

### Data is filtered, no more polishing required
#emp_counts = df['Employer'].value_counts()
#df2 = df[df['Employer'].isin(emp_counts[emp_counts > 3000].index)]
#job_counts = df2["Job Title"].value_counts()
#df2 = df2[df2["Job Title"].isin(job_counts[job_counts > 3000].index)]

FirAtt_lst = df2['Job Title'].unique()
SecAtt_lst = df2['Employer'].unique()
ThrAtt_lst = df2['Calendar Year'].unique()

#df2 = df.loc[df['Job Title'].isin(FirAtt_lst) & df['Employer'].isin(SecAtt_lst) & df['Calendar Year'].isin(ThrAtt_lst)]
#df2['Salary Paid'] = df2['Salary Paid'].apply(lambda x:x.split('.')[0].strip()).replace({'\$':'', ',':''}, regex=True)

FirAtt_Vec   = np.zeros(len(FirAtt_lst), dtype=np.int)
SecAtt_Vec   = np.zeros(len(SecAtt_lst), dtype=np.int)
ThrAtt_Vec   = np.zeros(len(ThrAtt_lst), dtype=np.int)

###################################     Forming a context   #######################################
Sal_outliers = np.array([1])
while(Sal_outliers[Sal_outliers.argmin()]==1):
	print '\n Looking for an original context \n'
	FirAtt_Vec[0:len(FirAtt_Vec)] = np.random.randint(2, size=len(FirAtt_Vec))
	SecAtt_Vec[0:len(SecAtt_Vec)] = np.random.randint(2, size=len(SecAtt_Vec))
	ThrAtt_Vec[0:len(ThrAtt_Vec)] = np.random.randint(2, size=len(ThrAtt_Vec))
	Orgn_Ctx = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(FirAtt_Vec== 1)].tolist()) & \
			   df2['Employer'].isin(SecAtt_lst[np.where(SecAtt_Vec== 1)].tolist()) & \
			   df2['Calendar Year'].isin(ThrAtt_lst[np.where(ThrAtt_Vec== 1)].tolist())]
#######################     Finding an outlier in the selected context      #######################
	clf = LocalOutlierFactor(n_neighbors=20)
	print '\n Sal_outliers is(before): \n',str(Sal_outliers)
	Sal_outliers = clf.fit_predict(Orgn_Ctx['Salary Paid'].values.reshape(-1,1))
  	print '\n Sal_outliers is(after): \n',str(Sal_outliers)
	
Queried_ID =Orgn_Ctx.iloc[Sal_outliers.argmin()][1]
print '\n\n Outlier\'s ID in the original context is: ', Queried_ID
# finding maximal context's size for queried_ID
max_ctx = maxctx(Ref_file, Queried_ID)
print '\nmaximal context has the population :\n', max_ctx

  ###########       Making Queue of samples and initiating it, with Org_Vec   ############################
Org_Vec       = np.zeros(len(FirAtt_Vec)+len(SecAtt_Vec)+len(ThrAtt_Vec))
np.concatenate((FirAtt_Vec, SecAtt_Vec, ThrAtt_Vec), axis=0, out=Org_Vec)
        ################################# Initiating queue with Org_ctx informaiton  ########################
Epsilon       = 0.001
Queue	      = [[0, np.exp(Epsilon *(Orgn_Ctx.shape[0])), Orgn_Ctx.shape[0], Org_Vec]]
Data_to_write = []
###################################        Flip the context ctx_Flpr(=100) times            ###############################
t0 = time.time()
BFS_Flp       = np.zeros(len(Org_Vec)) 
#Ctx_Flpr     = 0
Q_indx        = 0
index         = 0

while len(Queue)<100:     
    print '\nSampling & Queueing...  \n'
    for i in  range (len(Queue[Q_indx][3])):      
        BFS_Flp[i]  = Queue[Q_indx][3][i]
    while any(np.array_equal(BFS_Flp[:],x[3][:]) for x in Queue):
        for i in  range (len(Queue[Q_indx][3])):      
            BFS_Flp[i]  = Queue[Q_indx][3][i]
        Flp_bit           = random.randint(0,(len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)-1))
        BFS_Flp[Flp_bit]  = 1 - BFS_Flp[Flp_bit]
    BFS_Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(BFS_Flp[0:len(FirAtt_lst)-1] == 1)].tolist()) &\
		       df2['Employer'].isin(SecAtt_lst[np.where(BFS_Flp[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)-1] == 1)].tolist())  &\
		       df2['Calendar Year'].isin(ThrAtt_lst[np.where(BFS_Flp[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)-1] == 1)].tolist())]
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
                Score = np.exp(Epsilon *(BFS_Ctx.shape[0]))
                Queue.append([len(Queue), Score, BFS_Ctx.shape[0], np.zeros(len(Org_Vec))])
                for i in  range (len(Queue[len(Queue)-1][3])):      
                    Queue[len(Queue)-1][3][i]  = BFS_Flp[i]
                #print '\n Queue updated!'
    #print '\n Ctx_Flpr is = ', Ctx_Flpr, '\n The private context candidates are: \n', Queue
    ###################################       Sampling form the Queue ###############################
    elements = [elem[0] for elem in Queue]
    probabilities = [prob[1] for prob in Queue]/(sum ([prob[1] for prob in Queue]))
    ExpRes = np.random.choice(elements, 1, p = probabilities)
    for child in range(0, len(Queue)):
        if Queue[child][0] == ExpRes[0]:
            Q_indx = child      
    #Ctx_Flpr+=1
    Data_to_write.append(Queue[ Q_indx][2]/max_ctx) 

#print 'The candidate picked form the Q is ', ExpRes[0], 'th, with context ', Queue[ExpRes[0]][3][:],' and has ', Queue[ExpRes[0]][2], 'population'
t1 = time.time()
runtime = str(int((t1-t0) / 3600)) + ' hours and ' + str(int(((t1-t0) % 3600)/60)) + \
	' minutes and ' + str(((t1-t0) % 3600)%60) + ' seconds\n'
	    	   
writefinal(Data_to_write, str(int(sys.argv[1])), runtime, str(Queried_ID))	
#print '\n The final Queue is \n', Queue     
print '\n The BFS runtime, starting from org_ctx and choosing randomly one among childern in each layer is \n', runtime
