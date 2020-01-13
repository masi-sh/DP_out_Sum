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

random.seed(50*int(sys.argv[1]))
query_num = int(sys.argv[1])
Query_file = '/home/sm2shafi/DP_out_Sum/Murder/LOF/MLQueries_28.csv'
Queries = pd.read_csv(Query_file)
df2 = pd.read_csv("~/DP_out_Sum/dataset/MurderData_28.csv")
Ref_file = '/home/sm2shafi/DP_out_Sum/Murder/LOF/MLOFRef_28.txt'

def maxctx(Ref_file, ID):
	max  = 0
	size = 0
	outlier_ctr = 0
	with open(Ref_file,'rt') as f:
        	for num, line in enumerate(f, 1):
			ctx = line[1:-2].split(',')
                	size = int(ctx[3])
			for outliers in range(len(ctx)):
				if int(ctx[outliers])==ID:
   	        			outlier_ctr += 1
					if (max < size):
						max = size
					break				
	f.close()
	return max, outlier_ctr;

FirAtt_lst = df2['Weapon'].unique()
SecAtt_lst = df2['State'].unique()
ThrAtt_lst = df2['AgencyType'].unique()
FirAtt_Vec   = np.zeros(len(FirAtt_lst), dtype=np.int)
SecAtt_Vec   = np.zeros(len(SecAtt_lst), dtype=np.int)
ThrAtt_Vec   = np.zeros(len(ThrAtt_lst), dtype=np.int)
###################################     Forming a context   #######################################
Sal_outliers = np.array([1])
while(Sal_outliers[Sal_outliers.argmin()]==1):
	FirAtt_Vec[0:len(FirAtt_Vec)] = np.random.randint(2, size=len(FirAtt_Vec))
	SecAtt_Vec[0:len(SecAtt_Vec)] = np.random.randint(2, size=len(SecAtt_Vec))
	ThrAtt_Vec[0:len(ThrAtt_Vec)] = np.random.randint(2, size=len(ThrAtt_Vec))

	Orgn_Ctx = df2.loc[df2['Weapon'].isin(FirAtt_lst[np.where(FirAtt_Vec== 1)].tolist()) & \
			   df2['State'].isin(SecAtt_lst[np.where(SecAtt_Vec== 1)].tolist()) & \
			   df2['AgencyType'].isin(ThrAtt_lst[np.where(ThrAtt_Vec== 1)].tolist())]
#######################     Finding an outlier in the selected context      #######################
	if (Orgn_Ctx.shape[0] > 20):
		clf = LocalOutlierFactor(n_neighbors=20)
		Sal_outliers = clf.fit_predict(Orgn_Ctx['VictimAge'].values.reshape(-1,1))
Queried_ID =Orgn_Ctx.iloc[Sal_outliers.argmin()][0]
#print '\n\n Outlier\'s ID in the original context is: ', Queried_ID
max_ctx, count = maxctx(Ref_file, Queried_ID)
Ctx_Max = 0
  ###########       Making Queue of samples and initiating it, with Org_Vec   ############################
Org_Vec      = np.zeros(len(FirAtt_Vec)+len(SecAtt_Vec)+len(ThrAtt_Vec))
np.concatenate((FirAtt_Vec, SecAtt_Vec, ThrAtt_Vec), axis=0, out=Org_Vec)
#print '\n Org_Vec is: ' , Org_Vec

if (max_ctx !=0 and count>100):
	with open(Query_file, 'ab') as csvfile:
        	writer = csv.writer(csvfile)
        	fcntl.flock(csvfile, fcntl.LOCK_EX)
        	writer.writerow([query_num, Queried_ID, max_ctx, str(Org_Vec), Ctx_Max])
        	fcntl.flock(csvfile, fcntl.LOCK_UN)
	csvfile.close()
