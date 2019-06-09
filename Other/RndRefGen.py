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

df2 = pd.read_csv("~/DP_out_Sum/dataset/FilteredData.csv")
OutFile  = 'Outputs/output'+sys.argv[1]+'.txt'

def writefinal():	
	ff = open(OutFile,'a+')
	fcntl.flock(ff, fcntl.LOCK_EX)
	np.savetxt(ff, )
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

# Exploring Contexts and their outliers
for i in range (0, len(FirAtt_Sprset)):
	for j in range(int(sys.argv[1]), (int(sys.argv[1])+1)):
   		for z in range(0, len(ThrAtt_Sprset)):
            		print '\n\n The ', i, 'th element in the first attribute\'s superset'
            		print '\n\n The ', j, 'th element in the Second attribute\'s superset'
            		print '\n\n The ', z, 'th element in the third attribute\'s superset'
			print 'The percentage done: ', (2^(i+j+z))//2^25 
		
			Ctx  = df2.loc[df2['Job Title'].isin(FirAtt_Sprset[i]) & df2['Employer'].isin(SecAtt_Sprset[j]) &\
				       df2['Calendar Year'].isin(ThrAtt_Sprset[z])]
			for row in range(Ctx.shape[0]):
	                	


##########################3                                     ##########################


            print '\n\n Population size for orignial context plus first attribute set=', i, \
                    'Second attribute set=', j, 'Third attribute set=', z , 'is' , pop_size
#####################         Outlier detection in subpopulations      ########################
            if (pop_size >= 20):
                #Score = np.exp(Epsilon *(pop_size** (1. / 3)))
                Sal_arr= np.array(Sal_list)
                clf = LocalOutlierFactor(n_neighbors=20)
                Sal_outliers = clf.fit_predict(Sal_arr.reshape(-1,1))
		context.append([i,j,z,pop_size])
		outputfile.write('Context:\n'+str(context)+'\n')
		#print '\n\nlen(ID_list) is', len(ID_list)
		#print '\n\nSal_outliers for the context++ is', Sal_outliers, '\n\n An example of outlier here is', df2.iloc[Sal_outliers.argmin()][1]
                for outlier_finder in range(0, len(ID_list)):
                    if (Sal_outliers[outlier_finder]==-1): 
			output.append(ID_list[outlier_finder]) 
			if (ID_list[outlier_finder]==Queried_ID):                       
				Sub_pop.append([i,j,z,pop_size, Score, Sub_pop_count])
                        	Sub_pop_count += 1
		Sub_pop_sorted = sorted(Sub_pop,key=lambda Sub_pop: Sub_pop[3])
		
		#print '\n\nSubpopulations are[Att1_index, Att2_index, Population_size, Score, ID]\n\n', Sub_pop	
		print '\n\nSubpopulations sorted based on the population size are[Att1_index, Att2_index, Att2_index, Population_size, Score, ID]\n\n', \
                Sub_pop_sorted
		#print '\n\n str(output)', str(output)
		outputfile.write('ID_list:\n'+str(output)+'\n')

outputfile.close()
if Sub_pop_sorted:
	fcntl.flock(Maxfile, fcntl.LOCK_EX)
	Maxfile.write(str(Sub_pop_sorted[Sub_pop_count-1])+'\n')
        fcntl.flock(Maxfile, fcntl.LOCK_UN)	
	print 'Max_population is', Sub_pop_sorted[Sub_pop_count-1]   

Maxfile.close()

t1 = time.time()
print '\n\nThe required time for running the program is:',  t1-t0
