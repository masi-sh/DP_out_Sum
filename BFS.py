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
import random
#outputname  = 'Outputs/output'+sys.argv[1]+'.txt'
#Maxfilename = 'Max.txt'

##### Replace this with manual employer and job values to match with the go file output, this is a temporary fix, 
##### a long-term solution is fixing the go file
#emp_counts = df['Employer'].value_counts()
#df2 = df[df['Employer'].isin(emp_counts[emp_counts > 3000].index)]

#job_counts = df2["Job Title"].value_counts()
#df2 = df2[df2["Job Title"].isin(job_counts[job_counts > 3000].index)]

#FirAtt_lst = df2['Job Title'].unique()
#SecAtt_lst = df2['Employer'].unique()
FirAtt_lst = np.asarray(['Elementary Principal', 'Principal', 'Sergeant', 'Police Constable', 'Secondary Teacher', \
              'Assistant Professor', 'Firefighter', 'Teacher', 'Faculty Member', 'Professor', 'Constable', \
              'Detective', 'Associate Professor', 'Staff Sergeant', 'Plainclothes Police Constable', \
              'Senior Technical Engineer/Officer', 'Nuclear Operator', 'Registered Nurs'])
SecAtt_lst = np.asarray(['Peel District School Board', 'City of Ottawa - Police Services', 'Ryerson University', \
              'Dufferin-Peel Catholic District School Board', 'University of Western Ontario', 'University of Guelph', \
              'Community Safety & Correctional Services', 'Attorney General', 'McMaster University', \
              'City of Toronto - Police Service', 'University of Waterloo', 'Carleton University', 'York University', \
              'Ontario Power Generation', 'Regional Municipality of Peel - Police Services', \
              'York Region District School Board'])
ThrAtt_lst = df['Calendar Year'].unique()

df2 = df.loc[df['Job Title'].isin(FirAtt_lst) & df['Employer'].isin(SecAtt_lst) & df['Calendar Year'].isin(ThrAtt_lst)]
df2['Salary Paid'] = df2['Salary Paid'].apply(lambda x:x.split('.')[0].strip()).replace({'\$':'', ',':''}, regex=True)

FirAtt_Vec   = np.zeros(len(FirAtt_lst), dtype=np.int)
SecAtt_Vec   = np.zeros(len(SecAtt_lst), dtype=np.int)
ThrAtt_Vec   = np.zeros(len(ThrAtt_lst), dtype=np.int)

###################################     Forming a context   #######################################
FirAtt_Vec[0:5] = 1
SecAtt_Vec[0:6] = 1
ThrAtt_Vec[0:5] = 1
Orgn_Ctx = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(FirAtt_Vec== 1)].tolist()) & \
		   df2['Employer'].isin(SecAtt_lst[np.where(SecAtt_Vec== 1)].tolist()) & \
		   df2['Calendar Year'].isin(ThrAtt_lst[np.where(ThrAtt_Vec== 1)].tolist())]


Queue = Queue.append(Orgn_Ctx)

#######################     Finding an outlier in the selected context      #######################
clf = LocalOutlierFactor(n_neighbors=20)
Sal_outliers = clf.fit_predict(Orgn_Ctx['Salary Paid'].values.reshape(-1,1))
Queried_ID =Orgn_Ctx.iloc[Sal_outliers.argmin()][1]
print '\n\n Outlier\'s ID in the original context is: ', Queried_ID

        ############### Keeping attribute values in the original context, p =pr(1-->1)  ###############
Flp_p        = 0.7
       ############### Adding attribute values not in the original context, q =pr(0-->1) ###############
Flp_q        = 0.4
Flp_lst      = []
  ##############################        Making Queue of samples and initiating it       #################################
BFS_Vec      = np.zeros(len(FirAtt_Vec)+len(SecAtt_Vec)+len(ThrAtt_Vec))
np.concatenate((FirAtt_Vec, SecAtt_Vec, ThrAtt_Vec), axis=0, out=BFS_Vec)
        ################################# Initiating queue with Org_ctx informaiton  ########################
Queue	     = [[0, np.exp(Epsilon *(np.log(Org_Ctx.shape[0]))), Orgn_Ctx.shape[0], BFS_Vec]]
###################################        Flip the context ctx_Flpr(=100) times            ###############################
Epsilon = 0.1
Ctx_Flpr = 0
t0 = time.time()

while Ctx_Flpr<99:  
	
	BFS_Flp[:] = BFS_Vec[:]
	Flp_bit          = random.randint(0,43)
	BFS_Flp[Flp_bit] = 1 - BFS_Flp[Flp_bit]	
	BFS_Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(BFS_Flp[0:17]== 1)].tolist()) &\
			   df2['Employer'].isin(SecAtt_lst[np.where(BFS_Flp[18:33]== 1)].tolist()) &\
			   df2['Calendar Year'].isin(ThrAtt_lst[np.where(BFS_Flp[34:43]== 1)].tolist())]	
	
	Sal_list     = []
	ID_list      = []
	if (BFS_Ctx.shape[0] >= 20):
		for row in range(BFS_Ctx.shape[0]):
                    Sal_list.append(BFS_Ctx.iloc[row]['Salary Paid'])
		    ID_list.append(BFS_Ctx.iloc[row]['Unnamed: 0'])
			
                Score = np.exp(Epsilon *(np.log(BFS_Ctx.shape[0])))
                Sal_arr= np.array(Sal_list)
                clf = LocalOutlierFactor(n_neighbors=20)
                Sal_outliers = clf.fit_predict(Sal_arr.reshape(-1,1))
		for outlier_finder in range(0, len(ID_list)):
                    if ((Sal_outliers[outlier_finder]==-1) and (ID_list[outlier_finder]==Queried_ID)): 
			Queue.append([Ctx_Flpr+1, Score, BFS_Ctx.shape[0], BFS_Flp)
			print '\n Ctx_Flpr is = ', Ctx_Flpr, '\n The private context candidates are: \n', Queue
	###################################       Sampling form the Queue ###############################
			Q_smp  = len(Queue)
			elements = [elem[0] for elem in Queue]	
			probabilities = [prob[1] for prob in Queue]/(sum ([prob[1] for prob in Queue]))
			ExpRes = np.random.choice(elements, Q_smp, p = probabilities)
			BFS_Vec[:]  = Queue[[ExpRes][3][:]]
			Ctx_Flpr+=1
			print 'The candidate picked form the Q is ', ExpRes, 'th, with context ', Queue[[ExpRes][3]],\
				      ' and has ', Queue[[ExpRes][2]], 'population'
			
				      
				      
				      
       ###################################      Sampling form Exp Mech Result      #################################
num_smp  = 100
elements = [elem[0] for elem in Flp_lst]	
probabilities = [prob[1] for prob in Flp_lst]/(sum ([prob[1] for prob in Flp_lst]))
ExpRes = np.random.choice(elements, num_smp, p = probabilities)
print '\n\nThe number of candidates in Exponential mechanism range is:'           , len(Flp_lst)
print '\n\nIDs sampled from Exponential mechanism output are\n\n',  ExpRes

	#################################    Population size in the samples     #####################################
Flp_Ctx_sizes =[]  			
for ids in ExpRes:
	Flp_Ctx_sizes.append(Flp_lst[ids][2])
print '\n\nThe population size in the perturbed candidates are: \n\n', Flp_Ctx_sizes

t1 = time.time()
print '\n\nThe required time for running the program is:', t1-t0

	#	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%          FIXED UP TO HERE           %%%%%%%%5%%%%%%%%%%%%%%

	########################    Sample distance from outlier(in the number of attribute values)    ##############
print '\n\noutlier_index is: ', Queried_ID
#Smpl_out_dist =  [(len(Flp_lst[ids][3] - FirAtt_lst) + (len(Flp_lst[ids][4] - SecAtt_lst)\
#		+ (len(Flp_lst[ids][5] - ThrAtt_lst) for ids in ExpRes] 

#print '\n\nThe distance(in the number of attribute values) between perturbed candidates and the outlier is: \n\n', \
#       Smpl_out_dist

#t1 = time.time()
#print '\n\nThe required time for running the program is:',  t1-t0

plt.figure(1)
pd.Series(Flp_Ctx_sizes).value_counts().plot('bar')
#plt.figure(2)
#pd.Series(Smpl_out_dist).value_counts().plot('bar')
plt.show()
