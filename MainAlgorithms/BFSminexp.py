import matplotlib
matplotlib.use('Agg')
import sys
import pandas as pd
import numpy as np
import cufflinks as cf
import plotly
import plotly.offline as py
import plotly.graph_objs as go
cf.go_offline()
df = pd.read_csv("~/DP_out_Sum/dataset/combined.csv")
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
import time
import fcntl
import gzip
import random

################ Reference file to find the maximal context ###############
Ref_file = 'AllCTXOUT.txt.gz'
Store_file = 'BFSminexpDataPointsOutput.dat'

# Parallelizing using multiple cors, so each core needs to test just one data point 
Datapoints = 1
#outputname  = 'Outputs/output'+sys.argv[1]+'.txt'
#Maxfilename = 'Max.txt'

# To get the same original contexts in all files, each core gets seed from the bash file
#random.seed(2019)
random.seed(int(sys.argv[1]))

# Finds the maximal context for the Queried_ID      
def maxctx(Ref_file, Queried_ID):
	max = 0
	out_size = 0
	#line_num = 0
	size = 0
	#Ctx_line = 0
	with gzip.open(Ref_file,'rt') as f:
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
	#print "max so far is :", max, "in line number ", line_num
	f.close()
	return max;

# Writing final data 
def writefinal(Data_to_write, randomness, runtime, ID):	
	ff = open(Store_file,'a+')
	fcntl.flock(ff, fcntl.LOCK_EX)
	np.savetxt(ff, np.column_stack(Data_to_write), fmt=('%5i'), header = randomness+ ' Generates outlier , ' + ID + ', BFSexp alg. takes' + runtime)
	fcntl.flock(ff, fcntl.LOCK_UN)
	ff.close()
	return;

#### TO FIX: how to get the same number of output as the go file after filtering?
emp_counts = df['Employer'].value_counts()
df2 = df[df['Employer'].isin(emp_counts[emp_counts > 3000].index)]

job_counts = df2["Job Title"].value_counts()
df2 = df2[df2["Job Title"].isin(job_counts[job_counts > 3000].index)]

FirAtt_lst = df2['Job Title'].unique()
SecAtt_lst = df2['Employer'].unique()
ThrAtt_lst = df2['Calendar Year'].unique()

df2 = df2.loc[df2['Job Title'].isin(FirAtt_lst) & df2['Employer'].isin(SecAtt_lst) & df2['Calendar Year'].isin(ThrAtt_lst)]
df2['Salary Paid'] = df2['Salary Paid'].apply(lambda x:x.split('.')[0].strip()).replace({'\$':'', ',':''}, regex=True)

FirAtt_Vec   = np.zeros(len(FirAtt_lst), dtype=np.int)
SecAtt_Vec   = np.zeros(len(SecAtt_lst), dtype=np.int)
ThrAtt_Vec   = np.zeros(len(ThrAtt_lst), dtype=np.int)

###################################     Forming a context   #######################################
FirAtt_Vec[0:5] = 1
SecAtt_Vec[0:6] = 1
ThrAtt_Vec[0:5] = 1
FirAtt_Vec[5:len(FirAtt_Vec)] = np.random.randint(2, size=len(FirAtt_Vec)-5)
SecAtt_Vec[6:len(SecAtt_Vec)] = np.random.randint(2, size=len(SecAtt_Vec)-6)
ThrAtt_Vec[5:len(ThrAtt_Vec)] = np.random.randint(2, size=len(ThrAtt_Vec)-5)
Orgn_Ctx = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(FirAtt_Vec== 1)].tolist()) & \
		   df2['Employer'].isin(SecAtt_lst[np.where(SecAtt_Vec== 1)].tolist()) & \
		   df2['Calendar Year'].isin(ThrAtt_lst[np.where(ThrAtt_Vec== 1)].tolist())]

#######################     Finding an outlier in the selected context      #######################
clf = LocalOutlierFactor(n_neighbors=20)
Sal_outliers = clf.fit_predict(Orgn_Ctx['Salary Paid'].values.reshape(-1,1))
Queried_ID =Orgn_Ctx.iloc[Sal_outliers.argmin()][1]
print '\n\n Outlier\'s ID in the original context is: ', Queried_ID

############################      Minimal Context, and transfer vector as intermediate value in queueing     #############################################
mnml_Vec = np.zeros(len(FirAtt_Vec)+len(SecAtt_Vec)+len(ThrAtt_Vec))
mnml_Vec[np.where(FirAtt_lst == df2.values[Queried_ID][5])] = 1 
mnml_Vec[np.where(SecAtt_lst == df2.values[Queried_ID][4])[0]+len(FirAtt_lst)] = 1 
mnml_Vec[np.where(ThrAtt_lst == df2.values[Queried_ID][7])[0]+(len(FirAtt_lst)+len(SecAtt_lst))] = 1 
mnml_Ctx = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(mnml_Vec[0:len(FirAtt_lst)-1] == 1)].tolist()) &\
                   df2['Employer'].isin(SecAtt_lst[np.where(mnml_Vec[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)-1] == 1)].tolist())  &\
                   df2['Calendar Year'].isin(ThrAtt_lst[np.where(mnml_Vec[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)-1] == 1)].tolist())]

trsf_Vec = np.zeros(len(FirAtt_Vec)+len(SecAtt_Vec)+len(ThrAtt_Vec))
trsf_Vec[np.where(FirAtt_lst == df2.values[Queried_ID][5])] = 1 
trsf_Vec[np.where(SecAtt_lst == df2.values[Queried_ID][4])[0]+len(FirAtt_lst)] = 1 
trsf_Vec[np.where(ThrAtt_lst == df2.values[Queried_ID][7])[0]+(len(FirAtt_lst)+len(SecAtt_lst))] = 1
################################# Initiating queue with Minimal Context informaiton  ########################
Epsilon = 0.0001
Data_to_write = []
Min_Sal_list = []
Min_ID_list  = []
for row in range(mnml_Ctx.shape[0]):
	Min_Sal_list.append(mnml_Ctx.iloc[row]['Salary Paid'])
	Min_ID_list.append(mnml_Ctx.iloc[row]['Unnamed: 0'])
			
Min_Score = np.exp(Epsilon *(0))
Min_Sal_arr= np.array(Min_Sal_list)
clf = LocalOutlierFactor(n_neighbors=20)
Min_Sal_outliers = clf.fit_predict(Min_Sal_arr.reshape(-1,1))
for outlier_finder in range(0, len(Min_ID_list)):
	if ((Min_Sal_outliers[outlier_finder]==-1) and (Min_ID_list[outlier_finder]==Queried_ID)): 
		Min_Score = np.exp(Epsilon *(mnml_Ctx.shape[0]))
		
Queue	= [[0, Min_Score, mnml_Ctx.shape[0], mnml_Vec]]
###################################      Add to the minimal context ctx_Flpr(=100) times    ###############################

Ctx_Flpr = 0
BFS_Flp  = np.zeros(len(mnml_Vec))
t0       = time.time()

while Ctx_Flpr<99:  
	sub_q        = []
	flpd	     = []
	Sub_Sal_list = []
	Sub_ID_list  = []
	Q_Cpr    = np.zeros(len(mnml_Vec))
	
	for Flp_bit in range(0,(len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst))):
		Sub_Score = np.exp(Epsilon *(0))
		BFS_Flp[:]   = trsf_Vec[:]
		if BFS_Flp[Flp_bit] == 0:
			BFS_Flp[Flp_bit] = 1
			BFS_Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(BFS_Flp[0:len(FirAtt_lst)-1] == 1)].tolist()) &\
					   df2['Employer'].isin(SecAtt_lst[np.where(BFS_Flp[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)-1] == 1)].tolist())  &\
					   df2['Calendar Year'].isin(ThrAtt_lst[np.where(BFS_Flp[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)-1] == 1)].tolist())]
			if (BFS_Ctx.shape[0] > 20):
				for row in range(BFS_Ctx.shape[0]):
					Sub_Sal_list.append(BFS_Ctx.iloc[row]['Salary Paid'])
					Sub_ID_list.append(BFS_Ctx.iloc[row]['Unnamed: 0'])		
				Sub_Sal_arr= np.array(Sub_Sal_list)
				clf = LocalOutlierFactor(n_neighbors=20)
				Sub_Sal_outliers = clf.fit_predict(Sub_Sal_arr.reshape(-1,1))
				for outlier_finder in range(0, len(Sub_ID_list)):
					if ((Sub_Sal_outliers[outlier_finder]==-1) and (Sub_ID_list[outlier_finder]==Queried_ID)):
						Sub_Score = np.exp(Epsilon *(BFS_Ctx.shape[0]))
			flpd[:] = BFS_Flp[:]
            		sub_q.append([Flp_bit ,Sub_Score , BFS_Ctx.shape[0], flpd[:]])
			
	#######################       Sampling from sub_queue(sampling in each layer)        ##################################
	Sub_elements = [elem[0] for elem in sub_q]	
	Sub_probabilities = [prob[1] for prob in sub_q]/(sum ([prob[1] for prob in sub_q]))
	SubRes = np.random.choice(Sub_elements, 1, p = Sub_probabilities)
	### delete next line, we can store the sub-exp result directly in the queue, we dont need an intermediate bfs_flp 
	#BFS_Flp[:]  = sub_q[SubRes[0]][3][:]	
	for child in range(0, len(sub_q)):
		if sub_q[child][0] == SubRes[0]:
			Q_indx = child
	while not any(np.array_equal(sub_q[Q_indx][3][:],x[3]) for x in Queue):
        	Queue.append([Ctx_Flpr+1, sub_q[child][1], sub_q[child][2], sub_q[Q_indx][3][:]])
	
	print '\n Ctx_Flpr is = ', Ctx_Flpr, '\n The private context candidates are: \n', Queue
	###################################       Sampling form the Queue ###############################
	elements = [elem[0] for elem in Queue]	
	probabilities = [prob[1] for prob in Queue]/(sum ([prob[1] for prob in Queue]))
	ExpRes = np.random.choice(elements, 1, p = probabilities)
	trsf_Vec[:]  = Queue[ExpRes[0]][3][:]
	Ctx_Flpr+=1
	print 'The candidate picked form the Q is ', ExpRes[0], 'th, with context ', Queue[ExpRes[0]][3][:],\
	' and has ', Queue[ExpRes[0]][2], 'population'
	
	Data_to_write.append(Queue[ExpRes[0]][1]) 
	
###################################       Writing final data ###############################
t1 = time.time()
runtime = str(int((t1-t0) / 3600)) + ' hours and ' + str(int(((t1-t0) % 3600)/60)) + \
' minutes and ' + str(((t1-t0) % 3600)%60) + ' seconds\n'
	    	   
writefinal(Data_to_write, str(int(sys.argv[1])), runtime, str(Queried_ID))	

print '\n The final Queue is \n', Queue     
print '\n The BFS runtime, starting from org_ctx and using Exp among childern in each layer is \n', runtime
