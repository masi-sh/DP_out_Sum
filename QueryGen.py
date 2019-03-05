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
import csv

random.seed(100*int(sys.argv[1]))
query_num = int(sys.argv[1])
Query_file = '~/DP_out_Sum/Queries.csv'
Queries = pd.read_csv(Query_file)
df2 = pd.read_csv("~/DP_out_Sum/dataset/FilteredData.csv")
Ref_file = '/home/sm2shafi/Reffile.txt'

def maxctx(Ref_file, Queried_ID):
	max         = 0
	out_size    = 0
	#line_num   = 0
	size        = 0
	outlier_ctr = 0
	#Ctx_line   = 0
	with open(Ref_file,'rt') as f:
        	for num, line in enumerate(f, 1):
                	if line.split(' ')[0].strip()=="Matching":
                          #Ctx_line = num
                        	size = int((line.split(' '))[5].strip(':\n'))
			elif line.strip().startswith("ID"):
			        if line.split(' ')[3].strip('#')==str(Queried_ID):
					out_size = size
					outlier_ctr += 1
				#Valid_line = Ctx_line
                	if (max < out_size):
			        max = out_size
				#line_num = Valid_line 
	#print "max so far is :", max, "in line number ", line_num
	f.close()
  	return max, outlier_ctr;

FirAtt_lst = df2['Job Title'].unique()
SecAtt_lst = df2['Employer'].unique()
ThrAtt_lst = df2['Calendar Year'].unique()
FirAtt_Vec   = np.zeros(len(FirAtt_lst), dtype=np.int)
SecAtt_Vec   = np.zeros(len(SecAtt_lst), dtype=np.int)
ThrAtt_Vec   = np.zeros(len(ThrAtt_lst), dtype=np.int)
###################################     Forming a context   #######################################
Sal_outliers = np.array([1])
while(Sal_outliers[Sal_outliers.argmin()]==1):
	FirAtt_Vec[0:len(FirAtt_Vec)] = np.random.randint(2, size=len(FirAtt_Vec))
	SecAtt_Vec[0:len(SecAtt_Vec)] = np.random.randint(2, size=len(SecAtt_Vec))
	ThrAtt_Vec[0:len(ThrAtt_Vec)] = np.random.randint(2, size=len(ThrAtt_Vec))

	Orgn_Ctx = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(FirAtt_Vec== 1)].tolist()) & \
			   df2['Employer'].isin(SecAtt_lst[np.where(SecAtt_Vec== 1)].tolist()) & \
			   df2['Calendar Year'].isin(ThrAtt_lst[np.where(ThrAtt_Vec== 1)].tolist())]
#######################     Finding an outlier in the selected context      #######################
	if (len(Orgn_Ctx)!=0):
		clf = LocalOutlierFactor(n_neighbors=20)
		Sal_outliers = clf.fit_predict(Orgn_Ctx['Salary Paid'].values.reshape(-1,1))

Queried_ID =Orgn_Ctx.iloc[Sal_outliers.argmin()][1]
#print '\n\n Outlier\'s ID in the original context is: ', Queried_ID
max_ctx, count = maxctx(Ref_file, Queried_ID)

  ###########       Making Queue of samples and initiating it, with Org_Vec   ############################
Org_Vec      = np.zeros(len(FirAtt_Vec)+len(SecAtt_Vec)+len(ThrAtt_Vec))
np.concatenate((FirAtt_Vec, SecAtt_Vec, ThrAtt_Vec), axis=0, out=Org_Vec)
#print '\n Org_Vec is: ' , Org_Vec

if (max_ctx !=0 and count>1000):
	with open(Query_file, 'ab') as csvfile:
        	writer = csv.writer(csvfile)
        	fcntl.flock(csvfile, fcntl.LOCK_EX)
        	writer.writerow([query_num, Queried_ID, max_ctx, str(Org_Vec)])
        	fcntl.flock(csvfile, fcntl.LOCK_UN)
	csvfile.close()
