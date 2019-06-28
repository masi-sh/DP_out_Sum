from __future__ import division
import sys
#import gzip
import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter
import time
import fcntl
import random
import csv
import math
import hashlib
from outliers import smirnov_grubbs as grubbs

df2 = pd.read_csv("~/DP_out_Sum/Grubbs/ToyData.csv")
OutFile  = 'GrubbsRef.txt'

def writefinal(OutFile, outliers):
        ff = open(OutFile,'a+')
        fcntl.flock(ff, fcntl.LOCK_EX)
        for sub_list in range(len(outliers)):
                if outliers[sub_list][3]!=0:
                        ff.write(str(outliers[sub_list])+'\n')
        fcntl.flock(ff, fcntl.LOCK_UN)
        ff.close()
        return;

FirAtt_lst = df2['Job Title'].unique()
SecAtt_lst = df2['Employer'].unique()
ThrAtt_lst = df2['Calendar Year'].unique()

# Supersets for each attribute
FirAtt_Sprset = sum(map(lambda r: list(combinations(FirAtt_lst[0:], r)), range(1, len(FirAtt_lst[0:])+1)), [])
SecAtt_Sprset = sum(map(lambda r: list(combinations(SecAtt_lst[0:], r)), range(1, len(SecAtt_lst[0:])+1)), [])
ThrAtt_Sprset = sum(map(lambda r: list(combinations(ThrAtt_lst[0:], r)), range(1, len(ThrAtt_lst[0:])+1)), [])

t0 = time.time()
ctx_count = 0
outliers = []
# Exploring Contexts and their outliers
#for i in range (250, len(FirAtt_Sprset)):
i = int(sys.argv[1])
for j in range (0, len(SecAtt_Sprset)):
#for j in range(int(sys.argv[1]), (int(sys.argv[1])+1)):
        for z in range(0, len(ThrAtt_Sprset)):
                ctx_count+=1
                print 'count is:', ctx_count #, ' The percentage done: %', ctx_count//(2**25) 

                Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_Sprset[i]) & df2['Employer'].isin(SecAtt_Sprset[j]) &\
                               df2['Calendar Year'].isin(ThrAtt_Sprset[z])]
                outliers.append([i, j, z, Ctx.shape[0]])
                if (Ctx.shape[0]>20):
                        Salary = Ctx['Salary Paid']
                        IDs    = Ctx['Unnamed: 0.1']
                        grubbs_result = grubbs.max_test_indices(Salary, alpha=0.05)
                        for GOutlier in grubbs_result:
                                outliers[len(outliers)-1].append(IDs.values[GOutlier])

writefinal(OutFile, outliers)
t1 = time.time()
print '\n\nThe required time for running the program is:',  t1-t0
