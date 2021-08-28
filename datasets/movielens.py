"""
This code creates the BTL score vectors for the MovieLens 100K Dataset.
Link: https://grouplens.org/datasets/movielens/
Files Required: ml-100k.zip
"""
"""
Install the choix library to run this code. Some functions in the code borrow implementation from the library.
https://github.com/lucasmaystre/choix
"""
import numpy as np
import pandas as pd
import choix
np.set_printoptions(precision=3, suppress=True)

df = pd.read_fwf("D:\\Ranking\\Mark 9\\datasets\\movielens\\ml-100k\\u.data", header=None)
strings = np.array(df.values)
data = np.zeros((100000,4))
votes_per_movie = np.zeros(1682)
for i,s in enumerate(strings):
    data[i] = np.array(s[0].split('\t'), dtype=np.int)
    votes_per_movie[int(data[i][1]-1)] += 1

# IDs of Top 100 based on number of votes
movies = np.argsort(votes_per_movie)[::-1][0:100]
# New IDs of all movies
indices = {}
for i in range(100):
    indices[movies[i]] = i

# Dict of Dictionaries corresponding to each user containing movies and ratings by him
user_dict = {}
for i in range(100000):
    user_id = int(data[i][0]-1)
    movie_id = int(data[i][1]-1)
    rating = data[i][2]
    if movie_id in movies:
        if user_id in user_dict:
            user_dict[user_id][movie_id] = rating 
        else:
            user_dict[user_id] = {}
            user_dict[user_id][movie_id] = rating

# Y_ij = log(w_i) - log(w_j)
Y = np.zeros((100,100))
rate_count = np.zeros((100,100))

for i in range(943):
    votes = user_dict[i]
    size = len(votes)
    keys = list(user_dict[i].keys())
    for j in range(size):
        p = indices[keys[j]]
        for k in range(j+1,size):
            q = indices[keys[k]]
            Y[p][q] += votes[keys[j]] - votes[keys[k]]
            Y[q][p] += votes[keys[k]] - votes[keys[j]]
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