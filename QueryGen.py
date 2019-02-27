import sys
import pandas as pd
import numpy as np
import cufflinks as cf
cf.go_offline()
df = pd.read_csv("~/DP_out_Sum/dataset/combined.csv")
from itertools import combinations
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
import time
import fcntl
import random

