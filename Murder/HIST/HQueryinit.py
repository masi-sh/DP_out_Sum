import sys
import pandas as pd
import numpy as np

Query_file = '/home/sm2shafi/DP_out_Sum/Murder/HIST/MHQueries_28.csv'
Titles = np.array([['ind', 'Outlier', 'Max','Ctx', 'Max_Ctx']])
Queries = pd.DataFrame(data=Titles[1:,1:], columns=Titles[0,1:])
Queries.to_csv(Query_file)
