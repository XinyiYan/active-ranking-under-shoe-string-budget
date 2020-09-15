"""
This code creates the BTL score vectors for the Jester Joke Dataset 1.
Link: http://eigentaste.berkeley.edu/dataset/
Files Required: jester_dataset_1_1.zip, jester_dataset_1_2.zip, jester_dataset_1_3.zip
"""
"""
Install the choix library to run this code. Some functions in the code borrow implementation from the library.
https://github.com/lucasmaystre/choix
"""
import numpy as np
import pandas as pd
import choix
import tqdm
np.set_printoptions(precision=3, suppress=True)

df1 = pd.read_excel("D:\\Ranking\\Mark 9\\datasets\\jester\\Dataset1\\jester-data-1.xls", header=None)
df2 = pd.read_excel("D:\\Ranking\\Mark 9\\datasets\\jester\\Dataset1\\jester-data-2.xls", header=None)
df3 = pd.read_excel("D:\\Ranking\\Mark 9\\datasets\\jester\\Dataset1\\jester-data-3.xls", header=None)
df = pd.concat([df1,df2,df3])

Y = np.zeros((100,100))
rate_count = np.zeros((100,100))
num_votes = np.array(df.values[0:73421, 0:1])
data = np.array(df.values[0:73421, 1:151])

for i in tqdm.tqdm(range(73421)):
    votes = np.where(data[i] < 99)[0]
    size = np.size(votes)
    for j in range(size):
        p = votes[j]
        for k in range(j+1,size):
            q = votes[k]
            Y[p][q] += data[i][p] - data[i][q]
            Y[q][p] += data[i][q] - data[i][p]
            rate_count[p][q] += 1
            rate_count[q][p] += 1

rate_count[np.where(rate_count==0)] = 1
Y /= rate_count

ones = np.ones((100,100))
chain = ones/(ones + np.exp(Y))
chain -= np.diag(np.ones(100)*0.5)
chain -= np.diag(chain.sum(axis=1))
params = choix.utils.log_transform(choix.utils.statdist(chain))
estimates = np.exp(params)
estimates /= np.sum(estimates)
print(1000*estimates)