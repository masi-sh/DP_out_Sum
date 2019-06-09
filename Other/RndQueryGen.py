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

random.seed(50*int(sys.argv[1]))
query_num = int(sys.argv[1])
Query_file = '~/DP_out_Sum/dataset/RndQueries.csv'
Queries = pd.read_csv(Query_file)
df2 = pd.read_csv("~/DP_out_Sum/dataset/FilteredData.csv")
Ref_file = '/DP_out_Sum/Other/output.txt'

def maxctx(Ref_file, Queried_ID):
	max         = 0
	out_size    = 0
	line_num    = 0
	size        = 0
	outlier_ctr = 0
	Ctx_line    = 0
	Ctx_Max     = ''
	with open(Ref_file,'rt') as f:
        	for num, line in enumerate(f, 1):
                	if line.split(' ')[0].strip()=="Matching":
                          	Ctx_line = num
                        	size = int((line.split(' '))[5].strip(':\n'))
			elif line.strip().startswith("ID"):
			        if line.split(' ')[3].strip('#')==str(Queried_ID):
					out_size = size
					outlier_ctr += 1
					Valid_line = Ctx_line
                	if (max < out_size):
			        max = out_size
				line_num = Valid_line 
	#print "max so far is :", max, "in line number ", line_num
	f.close()
	# Ctx_Max not necessery now, I just put 0 for it
	Ctx_Max = 0
	#with open(Ref_file,'rt') as ff:
	#	print "\nMax context is wiht size", max ,"is:\n"
	#	for i, x in enumerate(ff):
	#		if i in range (line_num+1, line_num+4):
	#			print x
	#			Ctx_Max = Ctx_Max + x
	#ff.close()
  	return max, outlier_ctr, Ctx_Max;

def hash_calc(i, j, z, ID):
        hash_value = hashlib.md5(str(i+1000*j+1000000*z)+str(ID))
        hash_hex = hash_value.hexdigest()
        #:as_int = int(hash_hex[30:32],16)
        #return (as_int%128==0);
return (hash_hex[30:32] == '80' or hash_hex[30:32] == '00');

FirAtt_lst = df2['Job Title'].unique()
SecAtt_lst = df2['Employer'].unique()
ThrAtt_lst = df2['Calendar Year'].unique()

# Supersets for each attribute
FirAtt_Sprset = sum(map(lambda r: list(combinations(FirAtt_lst[0:], r)), range(1, len(FirAtt_lst[0:])+1)), [])
SecAtt_Sprset = sum(map(lambda r: list(combinations(SecAtt_lst[0:], r)), range(1, len(SecAtt_lst[0:])+1)), [])
ThrAtt_Sprset = sum(map(lambda r: list(combinations(ThrAtt_lst[0:], r)), range(1, len(ThrAtt_lst[0:])+1)), [])

outliers = []
while():
	i = np.random.randint(len(FirAtt_Sprset)-1)
	j = np.random.randint(len(SecAtt_Sprset)-1)
	z = np.random.randint(len(ThrAtt_Sprset)-1)
	Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_Sprset[i]) & df2['Employer'].isin(SecAtt_Sprset[j]) &\
		       df2['Calendar Year'].isin(ThrAtt_Sprset[z])]
	outliers.append([i, j, z, Ctx.shape[0]])
        if (Ctx.shape[0]>20):
        	for row in range(Ctx.shape[0]):
                	ID = Ctx.iloc[row, 0]
                        if hash_calc(i, j, z, ID):
				outliers.append([i, j, z, Ctx.shape[0], ID])
				break


	
	
	

FirAtt_lst  = df2['Job Title'].unique()
SecAtt_lst  = df2['Employer'].unique()
ThrAtt_lst  = df2['Calendar Year'].unique()
FirAtt_Vec   = np.zeros(len(FirAtt_lst), dtype=np.int)
SecAtt_Vec   = np.zeros(len(SecAtt_lst), dtype=np.int)
ThrAtt_Vec   = np.zeros(len(ThrAtt_lst), dtype=np.int)
#Forming a context
Sal_outliers = np.array([1])
while(Sal_outliers[Sal_outliers.argmin()]==1):
	FirAtt_Vec[0:len(FirAtt_Vec)] = np.random.randint(2, size=len(FirAtt_Vec))
	SecAtt_Vec[0:len(SecAtt_Vec)] = np.random.randint(2, size=len(SecAtt_Vec))
	ThrAtt_Vec[0:len(ThrAtt_Vec)] = np.random.randint(2, size=len(ThrAtt_Vec))

	Orgn_Ctx = df2.loc[df2['Job Title'].isin(FirAtt_lst[np.where(FirAtt_Vec== 1)].tolist()) & \
			   df2['Employer'].isin(SecAtt_lst[np.where(SecAtt_Vec== 1)].tolist()) & \
			   df2['Calendar Year'].isin(ThrAtt_lst[np.where(ThrAtt_Vec== 1)].tolist())]
#######################     Finding an outlier in the selected context      #######################
	if (Orgn_Ctx.shape[0] > 20):
		clf = LocalOutlierFactor(n_neighbors=20)
		Sal_outliers = clf.fit_predict(Orgn_Ctx['Salary Paid'].values.reshape(-1,1))
Queried_ID =Orgn_Ctx.iloc[Sal_outliers.argmin()][1]
#print '\n\n Outlier\'s ID in the original context is: ', Queried_ID
max_ctx, count, Ctx_Max = maxctx(Ref_file, Queried_ID)

  ###########       Making Queue of samples and initiating it, with Org_Vec   ############################
Org_Vec      = np.zeros(len(FirAtt_Vec)+len(SecAtt_Vec)+len(ThrAtt_Vec))
np.concatenate((FirAtt_Vec, SecAtt_Vec, ThrAtt_Vec), axis=0, out=Org_Vec)
#print '\n Org_Vec is: ' , Org_Vec

if (max_ctx !=0 and count>500):
	with open(Query_file, 'ab') as csvfile:
        	writer = csv.writer(csvfile)
        	fcntl.flock(csvfile, fcntl.LOCK_EX)
        	writer.writerow([query_num, Queried_ID, max_ctx, str(Org_Vec), Ctx_Max])
        	fcntl.flock(csvfile, fcntl.LOCK_UN)
	csvfile.close()
