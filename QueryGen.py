import sys
import pandas as pd
import numpy as np
import cufflinks as cf
cf.go_offline()
from itertools import combinations
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
import time
import fcntl
import random

random.seed(100*int(sys.argv[1]))
query_num = int(sys.argv[1])
Ref_file = '~/AllCTXOUT.txt.gz'
Query_file = '~/DP_out_Sum/MainAlgorithms/Queries.csv'
Queries = pd.read_csv("~/DP_out_Sum/MainAlgorithms/Queries.csv")
df = pd.read_csv("~/DP_out_Sum/dataset/combined.csv")

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

#### TO FIX: how to get the same number of output as the go file after filtering?
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


  ###########       Making Queue of samples and initiating it, with Org_Vec   ############################
Org_Vec      = np.zeros(len(FirAtt_Vec)+len(SecAtt_Vec)+len(ThrAtt_Vec))
np.concatenate((FirAtt_Vec, SecAtt_Vec, ThrAtt_Vec), axis=0, out=Org_Vec)

#Queries.loc[query_num]=[Queried_ID, maxctx(Ref_file, Queried_ID), Org_Vec]
#TEST, WITHOUT MAX
Queries.loc[query_num]=[Queried_ID, 0, Org_Vec]
