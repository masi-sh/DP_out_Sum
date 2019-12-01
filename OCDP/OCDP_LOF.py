# This code investigates whether removing a datapoint affects the list of valid contexts for a particular outlier or not. 

from __future__ import division
import sys
#import gzip
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

query_num = int(sys.argv[1])
df = pd.read_csv("~/DP_out_Sum/Grubbs/ToyData.csv")
Query_file = '/home/sm2shafi/DP_out_Sum/LOF/TLQueries.csv'
Queries = pd.read_csv(Query_file)
Ref_file = '/home/sm2shafi/DP_out_Sum/LOF/LOFRef.txt'


FirAtt_lst = df['Job Title'].unique()
SecAtt_lst = df['Employer'].unique()
ThrAtt_lst = df['Calendar Year'].unique()

# Supersets for each attribute
FirAtt_Sprset = sum(map(lambda r: list(combinations(FirAtt_lst[0:], r)), range(1, len(FirAtt_lst[0:])+1)), [])
SecAtt_Sprset = sum(map(lambda r: list(combinations(SecAtt_lst[0:], r)), range(1, len(SecAtt_lst[0:])+1)), [])
ThrAtt_Sprset = sum(map(lambda r: list(combinations(ThrAtt_lst[0:], r)), range(1, len(ThrAtt_lst[0:])+1)), [])

def org_ctx( , ):
  
  return;
  
def neighbor_ref(, ):
  
  return;
        
def neighbor_ctx(, ):
  
  return;   
        
def neighbors_compare(, ):
  
  return match_num;   
  
t0 = time.time()  
org_ctx( , )
neighbor_ref(, )
neighbor_ctx(, )
neighbors_compare(, )
t1 = time.time()
print '\n\nThe required time for running the program is:',  t1-t0
     
