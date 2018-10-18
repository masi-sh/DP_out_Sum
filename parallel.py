import sys
import pandas as pd
import numpy as np
import matplotlib
import cufflinks as cf
import plotly
import plotly.offline as py
import plotly.graph_objs as go
cf.go_offline()
df = pd.read_csv("~/DP_out_Sum/dataset/combined.csv")
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
import time
import fcntl

#outputname  = 'Outputs/output'+sys.argv[1]+'.txt'
Maxfilename = 'Max.txt'


emp_counts = df['Employer'].value_counts()
df2 = df[df['Employer'].isin(emp_counts[emp_counts > 3000].index)]

emp_counts = df2["Job Title"].value_counts()
df2 = df2[df2["Job Title"].isin(emp_counts[emp_counts > 3000].index)]
df2['Salary Paid'] = df2['Salary Paid'].apply(lambda x:x.split('.')[0].strip()).replace({'\$':'', ',':''}, regex=True)


FirAtt_lst = df2['Job Title'].unique()
SecAtt_lst = df2['Employer'].unique()
ThrAtt_lst = df2['Calendar Year'].unique()

###################################     Forming a context   #######################################
Orgn_Ctx = df2.loc[df2['Job Title'].isin([FirAtt_lst[0],FirAtt_lst[1],FirAtt_lst[2],FirAtt_lst[3], FirAtt_lst[4]]) & \
                   df2['Employer'].isin([SecAtt_lst[0],SecAtt_lst[1], SecAtt_lst[2],SecAtt_lst[3], SecAtt_lst[4], SecAtt_lst[5]]) & \
                   df2['Calendar Year'].isin([ThrAtt_lst[0],ThrAtt_lst[1],ThrAtt_lst[2],ThrAtt_lst[3],ThrAtt_lst[4]])]


#######################     Finding an outlier in the selected context      #######################
clf = LocalOutlierFactor(n_neighbors=20)
Sal_outliers = clf.fit_predict(Orgn_Ctx['Salary Paid'].values.reshape(-1,1))
Queried_ID =Orgn_Ctx.iloc[Sal_outliers.argmin()][1]

print '\n\n Outlier\'s ID in the selected context is: ', Queried_ID

################# Exploring Contexts larger than the original to find the maximal #################
FirAtt_Sprset = sum(map(lambda r: list(combinations(FirAtt_lst[5:], r)), range(1, len(FirAtt_lst[5:])+1)), [])
SecAtt_Sprset = sum(map(lambda r: list(combinations(SecAtt_lst[6:], r)), range(1, len(SecAtt_lst[6:])+1)), [])
ThrAtt_Sprset = sum(map(lambda r: list(combinations(ThrAtt_lst[5:], r)), range(1, len(ThrAtt_lst[5:])+1)), [])

Sub_pop        =  []
Sub_pop_count  =  0
Epsilon        =  0.1  ### Privacy Parameter
output         =  []
context        =  []

t0 = time.time()
#outputfile = open(outputname,'w')
Maxfile    = open(Maxfilename, 'a')

############################## Finds Maximal ##########################################
for i in range (0, len(FirAtt_Sprset)):
 for j in range(int(sys.argv[1]), (int(sys.argv[1])+1)):
   for z in range(0, len(ThrAtt_Sprset)):
            Sal_list   = []
            ID_list    = []
            pop_size   = 0
            Score      = 1
            #csvfile.seek(0)
            print '\n\n The ', i, 'th element in the first attribute\'s superset'
            print '\n\n The ', j, 'th element in the Second attribute\'s superset'
            print '\n\n The ', z, 'th element in the third attribute\'s superset'
            for row in range(df2.shape[0]):
		# print '\n\nrow is' , df2.iloc[row]
                # FirAtt referes to 'Job Title', which is array cell #5, 
                # SecAtt referes to 'Employer', which is array cell #4,
                # ThrAtt referes to 'Calendar Year', which is array cell #7,
                # isnt union1d(FirAtt_Sprset[i], FirAtt_lst[:5])= FirAtt_Sprset[i]+ FirAtt_lst[:5]
                if ((df2.iloc[row][5] in (np.union1d(FirAtt_Sprset[i], FirAtt_lst[:5]))) & \
                    (df2.iloc[row][4] in  (np.union1d(SecAtt_Sprset[j], SecAtt_lst[:6]))) & \
                     (df2.iloc[row][7] in (np.union1d(ThrAtt_Sprset[z], ThrAtt_lst[:5])))):
                    pop_size += 1
                    Sal_list.append(df2.iloc[row]['Salary Paid'])
		    ID_list.append(df2.iloc[row]['Unnamed: 0'])


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
		#outputfile.write('Context:\n'+str(context)+'\n')
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
#		outputfile.write('ID_list:\n'+str(output)+'\n')

#outputfile.close()
if Sub_pop_sorted:
	fcntl.flock(Maxfile, fcntl.LOCK_EX)
	Maxfile.write(str(Sub_pop_sorted[Sub_pop_count-1])+'\n')
        fcntl.flock(Maxfile, fcntl.LOCK_UN)	
	print 'Max_population is', Sub_pop_sorted[Sub_pop_count-1]   

Maxfile.close()

t1 = time.time()
print '\n\nThe required time for running the program is:',  t1-t0
