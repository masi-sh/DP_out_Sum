from __future__ import division
import sys
import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter
from sklearn.neighbors import LocalOutlierFactor
import time
import fcntl
import random
import csv
import math
import hashlib
from outliers import smirnov_grubbs as grubbs

df2 = pd.read_csv("~/DP_out_Sum/dataset/MurderData_28.csv")
OutFile  = 'MLOFRef_28.txt'

def writefinal(OutFile, outliers):
        ff = open(OutFile,'a+')
        fcntl.flock(ff, fcntl.LOCK_EX)
        for sub_list in range(len(outliers)):
                if outliers[sub_list][3]!=0:
                        ff.write(str(outliers[sub_list])+'\n')
        fcntl.flock(ff, fcntl.LOCK_UN)
        ff.close()
        return;

FirAtt_lst = df2['Weapon'].unique()
SecAtt_lst = df2['State'].unique()
ThrAtt_lst = df2['AgencyType'].unique()

# Supersets for each attribute
FirAtt_Sprset = sum(map(lambda r: list(combinations(FirAtt_lst[0:], r)), range(1, len(FirAtt_lst[0:])+1)), [])
SecAtt_Sprset = sum(map(lambda r: list(combinations(SecAtt_lst[0:], r)), range(1, len(SecAtt_lst[0:])+1)), [])
ThrAtt_Sprset = sum(map(lambda r: list(combinations(ThrAtt_lst[0:], r)), range(1, len(ThrAtt_lst[0:])+1)), [])

t0 = time.time()
ctx_count = 0
outliers  = []
# Exploring Contexts and their outliers
#for i in range (250, len(FirAtt_Sprset)):
i = int(sys.argv[1])
for j in range (0, len(SecAtt_Sprset)):
        for z in range(0, len(ThrAtt_Sprset)):
		Sub_Sal_list = []
		Sub_ID_list  = []
                Ctx  = df2.loc[df2['Weapon'].isin(FirAtt_Sprset[i]) & df2['State'].isin(SecAtt_Sprset[j]) &\
			       df2['AgencyType'].isin(ThrAtt_Sprset[z])]
                outliers.append([i, j, z, Ctx.shape[0]])
                if (Ctx.shape[0]>20):
                        for row in range(Ctx.shape[0]):
				#VictimAge is column 4 and the ID is on column 0
				Sub_Sal_list.append(Ctx.iloc[row,4])
				Sub_ID_list.append(Ctx.iloc[row,0])		
			Sub_Sal_arr= np.array(Sub_Sal_list)
			clf = LocalOutlierFactor(n_neighbors=20)
			Sub_Sal_outliers = clf.fit_predict(Sub_Sal_arr.reshape(-1,1))
			for outlier_finder in range(0, len(Sub_ID_list)):
				if ((Sub_Sal_outliers[outlier_finder]==-1)):
                                        outliers[len(outliers)-1].append(Sub_ID_list[outlier_finder])                    
writefinal(OutFile, outliers)
t1 = time.time()
print '\n\nThe required time for generating LOFRef.txt is:',  t1-t0
