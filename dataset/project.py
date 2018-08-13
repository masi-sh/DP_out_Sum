import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
import csvgen
#import bigfloat
#import csvgen_million

with open('outds.csv', 'r') as csvfile:
	outdsreader = csv.DictReader(csvfile)
	t0 = time.time()

	######################     Building the superset for each attribute   ###############################
 
	Queried_ID    = '96000'   ### Queried Outlier
	Epsilon       =  0.1	  ### Privacy Parameter
	num_smp	      = 1000      ### Number of samples from Exp Mechanism range
	FirAtt_lst    = ['Eng', 'Dr', 'Lawyer', 'Prof']   		    #### Values of the first attribute
	SecAtt_lst    = ['Westmount', 'Beechwood', 'Lakeshore', 'Uptown']   #### Values of the second attribute
	FirAtt_Sprset = sum(map(lambda r: list(combinations(FirAtt_lst, r)), range(1, len(FirAtt_lst)+1)), [])	
	SecAtt_Sprset = sum(map(lambda r: list(combinations(SecAtt_lst, r)), range(1, len(SecAtt_lst)+1)), [])
	print '\n\n\nSuperset of the first attribute is:  \n\n', FirAtt_Sprset
	print '\n\n\nSuperset of the second attribute is: \n\n', SecAtt_Sprset


	#############################      Creating subpopulations       ####################################

	Sub_pop        = []
	Sub_pop_count  = 0
	for i in range ( 0, len(FirAtt_Sprset)):
		for j in range(0, len(SecAtt_Sprset)):
			Sal_list   = []
			ID_list    = []
			pop_size   = 0
			Score      = 1
			csvfile.seek(0)
			for row in outdsreader:
				if ((row.values()[0] in FirAtt_Sprset[i]) & (row.values()[1] in SecAtt_Sprset[j])):
					pop_size += 1  
					Sal_list.append(row['Salary(K)'])
					ID_list.append(row['ID'])

	#####################         Outlier detection in subpopulations      #############################

			if (pop_size!= 0):
				#Score = np.exp(Epsilon *np.log(pop_size))    ### Score Calculation
				#Score = np.exp(Epsilon *(pop_size))				
				Score = np.exp(Epsilon *(pop_size** (1. / 3)))				
				Sal_arr= np.array(Sal_list)
				clf = LocalOutlierFactor(n_neighbors=4)
				Sal_outliers = clf.fit_predict(Sal_arr.reshape(-1,1))
				for outlier_finder in range(0, len(ID_list)):
					if ((ID_list[outlier_finder]==Queried_ID) & (Sal_outliers[outlier_finder]==-1)): 
						Sub_pop.append([i,j,pop_size, Score, Sub_pop_count])
						Sub_pop_count += 1			
	
	Sub_pop_sorted = sorted(Sub_pop,key=lambda Sub_pop: Sub_pop[2])
	print '\n\nSubpopulations are[Att1_index, Att2_index, Population_size, Score, ID]\n\n', Sub_pop	
	print '\n\nSubpopulations sorted based on the score are[Att1_index, Att2_index, Population_size, Score, ID]\n\n', \
               Sub_pop_sorted


	############          Max subpopulation wiht least number of attribute values for outlier        ###########
	
	outlier_index = len(Sub_pop)-1	
	while (Sub_pop_sorted[outlier_index-1][2] == Sub_pop_sorted[len(Sub_pop)-1][2]):
		outlier_index = outlier_index-1
	
	print '\n\nThe queried ID is outlier among population size:(true answer)', Sub_pop_sorted[outlier_index][2], \
	'\nOutlier features [i, j, pop_size, Score, ID] are =', Sub_pop_sorted[outlier_index]



	###################################      Sampling form Exp Mech Result      #################################

	elements = [elem[4] for elem in Sub_pop]	
	probabilities = [prob[3] for prob in Sub_pop]/(sum ([prob[3] for prob in Sub_pop]))
	ExpRes = np.random.choice(elements, num_smp, p = probabilities)
	print '\n\nThe number of candidates in Exponential mechanism range is:'           , len(Sub_pop_sorted)
	print '\n\nIDs sampled from Exponential mechanism output are\n\n',  ExpRes

	
	#################################    Population size in the samples     #####################################
	sub_pop_sizes =[]  			
	for ids in ExpRes:
		sub_pop_sizes.append(Sub_pop[ids][2])
	print '\n\nThe population size in the perturbing candidates are: \n\n', sub_pop_sizes


	########################    Sample distance from outlier(in the number of attribute values)    ##############
	print '\n\noutlier_index is: ', outlier_index
	Smpl_out_dist =  [(len(FirAtt_Sprset[Sub_pop[ids][0]]) - len(FirAtt_Sprset[Sub_pop_sorted[outlier_index][0]])+ \
		    	   len(SecAtt_Sprset[Sub_pop[ids][1]]) - len(SecAtt_Sprset[Sub_pop_sorted[outlier_index][1]])) \
		     	   for ids in ExpRes] 

	print '\n\nThe distance(in the number of attribute values) between perturbing candidates and the outlier is: \n\n', \
	       Smpl_out_dist

	t1 = time.time()
	print '\n\nThe required time for running the program is:',  t1-t0

	plt.figure(1)
	pd.Series(sub_pop_sizes).value_counts().plot('bar')
	plt.figure(2)
	pd.Series(Smpl_out_dist).value_counts().plot('bar')
	plt.show()


			




