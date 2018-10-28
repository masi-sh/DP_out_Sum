import matplotlib
matplotlib.use('Agg')
import sys
import pandas as pd
import numpy as np
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
#Maxfilename = 'Max.txt'


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

print '\n\n Outlier\'s ID in the original context is: ', Queried_ID

Flp_Ctx = pd.DataFrame()
        ############### Keeping attribute values in the original context, p =pr(1-->1)  ###############
Flp_p        = 0.7
         ############### Adding attribute values not in the original context, q =pr(0-->1) ###############
Flp_q        = 0.4
FirAtt_Flp   = np.zeros(len(FirAtt_lst), dtype=np.int)
SecAtt_Flp   = np.zeros(len(SecAtt_lst), dtype=np.int)
ThrAtt_Flp   = np.zeros(len(ThrAtt_lst), dtype=np.int)
Flp_lst      = []
Sal_list     = []
ID_list      = []

###################################        Flip the context ctx_Flpr(=100) times            ###############################
Epsilon = 0.1
Ctx_Flpr = 0
while Ctx_Flpr<100:
	##### context separator scans all elements in the attribute lists to find where to apply p or q #######
	for Ctx_sprt in range (0, len(FirAtt_lst)):
		if ((Ctx_sprt<5 and np.random.binomial(size=1, n=1, p= Flp_p)==1) or \
		    (Ctx_sprt>=5 and np.random.binomial(size=1, n=1, p= Flp_q)==1)):
			FirAtt_Flp[Ctx_sprt]=1

	for Ctx_sprt in range (0, len(SecAtt_lst)):
		if ((Ctx_sprt<5 and np.random.binomial(size=1, n=1, p= Flp_p)==1) or \
		    (Ctx_sprt>=5 and np.random.binomial(size=1, n=1, p= Flp_q)==1)):
			SecAtt_Flp[Ctx_sprt]=1
	
	for Ctx_sprt in range (0, len(ThrAtt_lst)):
		if ((Ctx_sprt<5 and np.random.binomial(size=1, n=1, p= Flp_p)==1) or \
		    (Ctx_sprt>=5 and np.random.binomial(size=1, n=1, p= Flp_q)==1)):
			ThrAtt_Flp[Ctx_sprt]=1
	
	Flp_Ctx= Flp_Ctx.append(df2[(df2['Job Title'].isin(FirAtt_lst[np.where(FirAtt_Flp == 1)])) & \
				    (df2['Employer'].isin(SecAtt_lst[np.where(SecAtt_Flp == 1)])) & \
				    (df2['Calendar Year'].isin(ThrAtt_lst[np.where(ThrAtt_Flp == 1)]))])
	
	if (Flp_Ctx.shape[0] >= 20):
		for row in range(Flp_Ctx.shape[0]):
                    Sal_list.append(Flp_Ctx.iloc[row]['Salary Paid'])
		    ID_list.append(Flp_Ctx.iloc[row]['Unnamed: 0'])
                Score = np.exp(Epsilon *(np.log(Flp_Ctx.shape[0])))
                Sal_arr= np.array(Sal_list)
                clf = LocalOutlierFactor(n_neighbors=20)
                Sal_outliers = clf.fit_predict(Sal_arr.reshape(-1,1))
		for outlier_finder in range(0, len(ID_list)):
                    if ((Sal_outliers[outlier_finder]==-1) and (ID_list[outlier_finder]==Queried_ID)):  
			Flp_lst.append([Ctx_Flpr, Score, Flp_Ctx.shape[0], FirAtt_Flp, SecAtt_Flp, ThrAtt_Flp])
			print '\n Ctx_Flpr is = ', Ctx_Flpr, '\n The private context candidates are: \n',Flp_lst 
			Ctx_Flpr+=1
			
       ###################################      Sampling form Exp Mech Result      #################################
num_smp  = 100
elements = [elem[0] for elem in Flp_lst]	
probabilities = [prob[1] for prob in Flp_lst]/(sum ([prob[1] for prob in Flp_lst]))
ExpRes = np.random.choice(elements, num_smp, p = probabilities)
print '\n\nThe number of candidates in Exponential mechanism range is:'           , len(Flp_lst)
print '\n\nIDs sampled from Exponential mechanism output are\n\n',  ExpRes

	#################################    Population size in the samples     #####################################
Flp_Ctx_sizes =[]  			
for ids in ExpRes:
	Flp_Ctx_sizes.append(Flp_lst[ids][2])
print '\n\nThe population size in the perturbed candidates are: \n\n', Flp_Ctx_sizes


	#	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%          FIXED UP TO HERE           %%%%%%%%5%%%%%%%%%%%%%%

	########################    Sample distance from outlier(in the number of attribute values)    ##############
print '\n\noutlier_index is: ', Queried_ID
#Smpl_out_dist =  [(len(Flp_lst[ids][3] - FirAtt_lst) + (len(Flp_lst[ids][4] - SecAtt_lst)\
#		+ (len(Flp_lst[ids][5] - ThrAtt_lst) for ids in ExpRes] 

#print '\n\nThe distance(in the number of attribute values) between perturbed candidates and the outlier is: \n\n', \
#       Smpl_out_dist

#t1 = time.time()
#print '\n\nThe required time for running the program is:',  t1-t0

plt.figure(1)
pd.Series(Flp_Ctx_sizes).value_counts().plot('bar')
#plt.figure(2)
#pd.Series(Smpl_out_dist).value_counts().plot('bar')
plt.show()

			
		
