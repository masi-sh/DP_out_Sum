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

# To get the same original contexts in all files
random.seed(100*int(sys.argv[1]))

Ref_file = 'AllCTXOUT.txt.gz'
Store_file = 'USampleDataPointsOutput.dat'

# Writing final data 
def writefinal(Data_to_write, randomness, runtime, ID):	
	ff = open(Store_file,'a+')
	fcntl.flock(ff, fcntl.LOCK_EX)
	np.savetxt(ff, np.column_stack(Data_to_write), fmt=('%5i'), header = randomness+ ' Generates outlier , ' + ID + ', BFSexp alg. takes' + runtime)
	fcntl.flock(ff, fcntl.LOCK_UN)
	ff.close()
	return;

emp_counts = df['Employer'].value_counts()
df2 = df[df['Employer'].isin(emp_counts[emp_counts > 3000].index)]

job_counts = df2["Job Title"].value_counts()
df2 = df2[df2["Job Title"].isin(job_counts[job_counts > 3000].index)]

FirAtt_lst = df2['Job Title'].unique()
SecAtt_lst = df2['Employer'].unique()
ThrAtt_lst = df['Calendar Year'].unique()

df2 = df.loc[df['Job Title'].isin(FirAtt_lst) & df['Employer'].isin(SecAtt_lst) & df['Calendar Year'].isin(ThrAtt_lst)]
df2['Salary Paid'] = df2['Salary Paid'].apply(lambda x:x.split('.')[0].strip()).replace({'\$':'', ',':''}, regex=True)

FirAtt_Vec   = np.zeros(len(FirAtt_lst), dtype=np.int)
SecAtt_Vec   = np.zeros(len(SecAtt_lst), dtype=np.int)
ThrAtt_Vec   = np.zeros(len(ThrAtt_lst), dtype=np.int)

###################################     Forming a context   #######################################
FirAtt_Vec[0:5]=1
SecAtt_Vec[0:6]=1
ThrAtt_Vec[0:5]=1
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
Org_Vec      = np.zeros(len(FirAtt_Vec)+len(SecAtt_Vec)+len(ThrAtt_Vec))
np.concatenate((FirAtt_Vec, SecAtt_Vec, ThrAtt_Vec), axis=0, out=Org_Vec)

        ############### The probability of dding an attribute value to the context  ###############
Flp_p        = 0.5
Flp_lst      = []
Data_to_write = []
###################################        Flip the context ctx_Flpr(=100) times            ###############################
Epsilon = 0.001
#Ctx_Flpr = 0

t0 = time.time()
while len(Flp_lst)<100:
	
	Ctx_Flp = np.zeros(len(Org_Vec), dtype=np.int) 
	for Ctx_sprt in range (0, len(Ctx_Flp)):
        	if (np.random.binomial(size=1, n=1, p= Flp_p)==1):
                	Ctx_Flp[Ctx_sprt]=1
			
	Flp_Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(Ctx_Flp[0:len(FirAtt_lst)-1] == 1)].tolist()) &\
		       df2['Employer'].isin(SecAtt_lst[np.where(Ctx_Flp[len(FirAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)-1] == 1)].tolist())  &\
		       df2['Calendar Year'].isin(ThrAtt_lst[np.where(Ctx_Flp[len(FirAtt_lst)+len(SecAtt_lst):len(FirAtt_lst)+len(SecAtt_lst)+len(ThrAtt_lst)-1] == 1)].tolist())]
	
	Sal_list     = []
	ID_list      = []
	if (Flp_Ctx.shape[0] >= 20):
		for row in range(Flp_Ctx.shape[0]):
                    Sal_list.append(Flp_Ctx.iloc[row]['Salary Paid'])
		    ID_list.append(Flp_Ctx.iloc[row]['Unnamed: 0'])
                Score = np.exp(Epsilon *(Flp_Ctx.shape[0]))
                Sal_arr= np.array(Sal_list)
                clf = LocalOutlierFactor(n_neighbors=20)
                Sal_outliers = clf.fit_predict(Sal_arr.reshape(-1,1))
		for outlier_finder in range(0, len(ID_list)):
                    if ((Sal_outliers[outlier_finder]==-1) and (ID_list[outlier_finder]==Queried_ID)):  
			Flp_lst.append([len(Flp_lst), Score, Flp_Ctx.shape[0], np.zeros(len(Org_Vec))])
			print '\n Ctx_Flpr is = ', len(Flp_lst), '\n The private context candidates are: \n',Flp_lst
                	for i in  range (len(Flp_lst[len(Flp_lst)-1][3])):      
				Flp_lst[len(Flp_lst)-1][3][i] = Ctx_Flp[i]
				#Ctx_Flpr+=1

       ###################################      Sampling form Exp Mech Result      #################################
num_smp  = 100
elements = [elem[0] for elem in Flp_lst]	
probabilities = [prob[1] for prob in Flp_lst]/(sum ([prob[1] for prob in Flp_lst]))
ExpRes = np.random.choice(elements, num_smp, p = probabilities)
print '\n\nThe number of candidates in Exponential mechanism range is:'           , len(Flp_lst)
print '\n\nIDs sampled from Exponential mechanism output are\n\n',  ExpRes

for ids in ExpRes:
	Data_to_write.append(Flp_lst[ids][2]) 

t1 = time.time()
runtime = str(int((t1-t0) / 3600)) + ' hours and ' + str(int(((t1-t0) % 3600)/60)) + \
	' minutes and ' + str(((t1-t0) % 3600)%60) + ' seconds\n'
	    	   
writefinal(Data_to_write, str(int(sys.argv[1])), runtime, str(Queried_ID)) 
print '\n\nThe required time for running the Uniform Sampling algorithm is:', runtime

