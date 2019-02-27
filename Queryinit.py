import sys
import pandas as pd
import numpy as np
#import cufflinks as cf
#cf.go_offline()
#df = pd.read_csv("~/DP_out_Sum/dataset/combined.csv")
#from itertools import combinations
#from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
import time
import fcntl
import random

Titles = [['', 'Outlier', 'Ctx', 'Max']]
for i in range(NumofQueries):
  Titles.append([i, 0, str(np.zeros()), 0])
   

Queries = pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:])

Queries.to_csv(Store_file)
