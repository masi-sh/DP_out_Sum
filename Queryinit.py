import sys
import pandas as pd
import numpy as np

Query_file = '~/DP_out_Sum/MainAlgorithms/Queries.csv'
Titles = np.array([['index', 'Outlier', 'Max','Ctx']])
Queries = pd.DataFrame(data=data[1:,1:], columns=data[0,1:])
Queries.to_csv(Query_file)
