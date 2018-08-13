
############### Data From: https://www.ontario.ca/page/public-sector-salary-disclosure#section-0 ############

import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
#import csvgen
#import bigfloat
#import csvgen_million

df = pd.read_csv('/home/sm2shafi/Downloads/Salary Data/combined.csv')
t0 = time.time()

	######################     Building the superset for each attribute   ###############################
 
Queried_ID    = 96000   ### Queried Outlier
Epsilon       =  0.1	  ### Privacy Parameter
num_smp	      = 1000      ### Number of samples from Exp Mechanism range
FirAtt        = df['Employer'].unique().tolist()  		    #### Values of the first attribute
SecAtt        = df['Job Title'].unique().tolist() 		    #### Values of the first attribute
ThrAtt        = df['Calendar Year'].unique().tolist()  		    #### Values of the first attribute
FirAtt        = filter(lambda x : x != df.iloc[Queried_ID][4], FirAtt)
SecAtt        = filter(lambda x : x != df.iloc[Queried_ID][5], SecAtt)
ThrAtt        = filter(lambda x : x != df.iloc[Queried_ID][7], ThrAtt)

print '\n\n First Attribute is: ', FirAtt
print '\n\n Second Attribute is:', SecAtt
print '\n\n Third Attribute is: ', ThrAtt

ThrAtt_Sprset = sum(map(lambda r: list(combinations(ThrAtt, r)), range(1, len(ThrAtt)+1)), [])
print '\n\n Third Attributes Superset is: ', ThrAtt_Sprset

FirAtt_Sprset = sum(map(lambda r: list(combinations(FirAtt, r)), range(1, len(FirAtt)+1)), [])	
SecAtt_Sprset = sum(map(lambda r: list(combinations(SecAtt, r)), range(1, len(SecAtt)+1)), [])

print '\n\n First Attributes Superset is: ', FirAtt_Sprset
print '\n\n Second Attributes Superset is:', SecAtt_Sprset


t1 = time.time()
print '\n\nThe required time for running the program is:',  t1-t0
	#############################      Creating subpopulations       ####################################

