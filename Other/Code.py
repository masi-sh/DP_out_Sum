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


emp_counts = df['Employer'].value_counts()
df2 = df[df['Employer'].isin(emp_counts[emp_counts > 3000].index)]

emp_counts = df2["Job Title"].value_counts()
df2 = df2[df2["Job Title"].isin(emp_counts[emp_counts > 3000].index)]
df2['Salary Paid'] = df2['Salary Paid'].apply(lambda x:x.split('.')[0].strip()).replace({'\$':'', ',':''}, regex=True)


FirAtt_lst = df2['Job Title'].unique()
SecAtt_lst = df2['Employer'].unique()
ThrAtt_lst = df2['Calendar Year'].unique()

FirAtt_Sprset = sum(map(lambda r: list(combinations(FirAtt_lst, r)), range(1, len(FirAtt_lst)+1)), [])
SecAtt_Sprset = sum(map(lambda r: list(combinations(SecAtt_lst, r)), range(1, len(SecAtt_lst)+1)), [])
ThrAtt_Sprset = sum(map(lambda r: list(combinations(ThrAtt_lst, r)), range(1, len(ThrAtt_lst)+1)), [])

Queried_ID    = '2'
 
Sub_pop        = []
Sub_pop_count  = 0
Epsilon       =  0.1  ### Privacy Parameter

t0 = time.time()

for i in range ( 0, len(FirAtt_Sprset)):
 for j in range(0, len(SecAtt_Sprset)):
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
               # FirAtt referes to 'Job Title', which is array cell #5, 
               # SecAtt referes to 'Employer', which is array cell #4,
               # ThrAtt referes to 'Calendar Year', which is array cell #7,
                if ((df2.iloc[row][5] in FirAtt_Sprset[i]) & (df2.iloc[row][4] in SecAtt_Sprset[j]) & (df2.iloc[row][7] in ThrAtt_Sprset[z])):
                    pop_size += 1
                    Sal_list.append(df2.iloc[row]['Salary Paid'])
                    ID_list.append(df2.iloc[row]['Unnamed: 0'])
            print '\n\n Population size for first attribute set=', i, 'Second attribute set=', j, 'Third attribute set=', z , 'is' , pop_size    

#####################         Outlier detection in subpopulations      ########################
            if (pop_size >= 4):
                Score = np.exp(Epsilon *(pop_size** (1. / 3)))
                Sal_arr= np.array(Sal_list)
                clf = LocalOutlierFactor(n_neighbors=4)
                Sal_outliers = clf.fit_predict(Sal_arr.reshape(-1,1))
                for outlier_finder in range(0, len(ID_list)):
                    if ((ID_list[outlier_finder]==Queried_ID) & (Sal_outliers[outlier_finder]==-1)): 
                        Sub_pop.append([i,j,pop_size, Score, Sub_pop_count])
                        Sub_pop_count += 1
print '\n\nSubpopulations are[Att1_index, Att2_index, Population_size, Score, ID]\n\n', Sub_pop	
 
t1 = time.time()
print '\n\nThe required time for running the program is:',  t1-t0
