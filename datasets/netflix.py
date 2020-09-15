"""
This code creates the BTL score vectors for the Netflix Prize data.
Offical Link: https://www.netflixprize.com/
Data Link: https://www.kaggle.com/netflix-inc/netflix-prize-data
Files Required: entire training_set and associated files
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

ratings_per_movie = np.zeros(17770)
for i in tqdm.tqdm(range(17770), desc="ratings"):
    df = pd.read_csv("D:\\Ranking\\Mark 9\\datasets\\netflix\\training_set\\mv_%07d.txt" % (i+1), skiprows=[0], header=None)
    ratings_per_movie[i] = len(df.values)

df = pd.read_csv("D:\\Ranking\\Mark 9\\datasets\\netflix\\ratings_per_movie.csv")
ratings_per_movie = np.array(df['number of ratings'], dtype=np.int)

# IDs of Top 100 based on number of votes
movies = np.argsort(ratings_per_movie)[::-1][0:100]
# New IDs of all movies
indices = {}
for i in range(100):
    indices[movies[i]] = i

user_dict = {}
for movie_id in tqdm.tqdm(movies, desc="movies"):
    df = pd.read_csv("D:\\Ranking\\Mark 9\\datasets\\netflix\\training_set\\mv_%07d.txt" % (movie_id+1), skiprows=[0], header=None)
    data = np.array(df.values[0:len(df.values)])
    for i in range(len(data)):
        user_id = int(data[i][0])
        rating = data[i][1]
        if user_id in user_dict:
            user_dict[user_id][movie_id] = rating 
        else:
            user_dict[user_id] = {}
            user_dict[user_id][movie_id] = rating

# Y_ij = log(w_i) - log(w_j)
Y = np.zeros((100,100))
rate_count = np.zeros((100,100))

for i in tqdm.tqdm(list(user_dict.keys()), desc="uesers"):
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