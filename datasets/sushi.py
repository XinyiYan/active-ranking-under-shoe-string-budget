"""
This code creates the BTL score vectors for the SUSHI Preference Data Sets.
Link: http://www.kamishima.net/sushi/
Files Required: sushi3-2016.zip
"""
"""
Install the choix library to run this code. Some functions in the code borrow implementation from the library.
https://github.com/lucasmaystre/choix
"""
import numpy as np
import pandas as pd
import choix
np.set_printoptions(precision=3, suppress=True)

df1 = pd.read_fwf("D:\\Ranking\\Mark 9\\datasets\\sushi\\sushi3a.txt", header=None)
orders1 = np.array(df1.values[0:5000, 2:12])

data_A = []
for i in range(5000):
    for j in range(10):
        for k in range(j+1, 10):
            data_A.append((orders1[i][j], orders1[i][k]))
            
params_A = choix.rank_centrality(10, data_A)
estimates_A = np.exp(params_A)
estimates_A /= np.sum(estimates_A)
print(estimates_A)

df2 = pd.read_fwf("D:\\Ranking\\Mark 9\\datasets\\sushi\\sushi3b.txt", header=None)
strings = df2.values[0:5000, 2:3]
orders2 = [np.array(s[0].split(), dtype=np.int) for s in strings]

data_B = []
for i in range(5000):
    for j in range(10):
        for k in range(j+1, 10):
            data_B.append((orders2[i][j], orders2[i][k]))
            
params_B = choix.rank_centrality(100, data_B)
estimates_B = np.exp(params_B)
estimates_B /= np.sum(estimates_B)
print(100*estimates_B)