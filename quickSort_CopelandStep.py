import numpy as np
import random
import time

np.set_printoptions(precision=3, suppress=True)

global n_items
n_items = 50

global initial
initial = n_items

global tiks
tiks = 20

global Budget
Budget = n_items + initial

global count 
#count = 0

global ranking
ranking = []

global ranks
ranks = []

global data
#data = []

global comp_matrix
#comp_matrix = np.zeros(((n_items+1), (n_items+1)))

global scores
global true_top
scores = 75*np.random.rand(n_items)
#for i in range(n_items):
#    scores[i] = i+1
#random.shuffle(scores)
#scores = np.array([ 2.637, 37.092, 35.992,  4.858, 23.636, 30.448, 45.242, 30.948,
#                   29.136, 38.14 , 34.818, 29.57 , 30.447, 24.912, 28.451, 74.238,
#                   73.627, 26.471, 35.294, 36.311, 70.924, 39.424, 64.263,  3.016, 40.923])
scores = np.array([69.72 , 66.96 , 59.993, 53.81 , 44.211, 43.364, 49.359, 12.563, 34.298, 49.05 , 14.32 , 45.721, 60.111, 37.027, 66.176, 47.118,
                   60.457, 38.228, 15.522, 65.783, 65.161, 39.038, 73.657, 52.103, 33.216, 65.346, 36.982, 44.408, 64.675, 40.606,  0.199, 42.142,
                   65.422, 69.91 , 29.524,  8.012, 17.244, 70.475, 57.783, 51.637, 46.264, 46.619,  8.721, 39.563, 53.239, 49.05 , 26.752, 37.529, 60.218, 44.554])
#scores = np.array([  8.2  , 100.   ,  48.174,  42.926,  52.671,  64.874,  27.607,
#                   62.456,  64.792,  26.614])
true_top = np.argmax(scores)
scores[true_top] = 100
'''
scores = np.zeros(n_items + 1)
for i in range(n_items + 1):
    scores[i] = random.uniform(0,75)
scores = np.sort(scores)
scores[n_items] = 100
#'''

global diff
diff = []

def compare(a, b):
    global count
    if(count < Budget):
        count += 1
        if(random.uniform(0, scores[a-1]+scores[b-1]) < scores[a-1]):
            comp_matrix[a][b] += 1
            return False
        else:
            comp_matrix[b][a] += 1
            return True
    else:
        count += 1
        if(random.uniform(0, 1) < 0.5):
            comp_matrix[a][b] += 1
            return False
        else:
            comp_matrix[b][a] += 1
            return True

# This function takes last element as pivot, places 
# the pivot element at its correct position in sorted 
# array, and places all smaller (smaller than pivot) 
# to left of pivot and all greater elements to right 
# of pivot 
def partition(arr,low,high): 
    i = ( low-1 )         # index of smaller element 
    pivot = arr[random.randint(low, high)]     # pivot 
  
    for j in range(low , high): 
  
        # If current element is smaller than the pivot 
        if compare(arr[j], pivot): 
          
            # increment index of smaller element 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return ( i+1 ) 
  
# The main function that implements QuickSort 
# arr[] --> Array to be sorted, 
# low  --> Starting index, 
# high  --> Ending index 
  
# Function to do Quick sort 
def quickSort(arr,low,high): 
    if low < high: 
  
        # pi is partitioning index, arr[p] is now 
        # at right place 
        pi = partition(arr,low,high) 
  
        # Separately sort elements before 
        # partition and after partition 
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high)

def get_ranks(scores):
    ranks = np.zeros(len(scores))
    tmp = [(scores[i], i) for i in range(len(scores))]
    tmp.sort(key=lambda x: x[0], reverse = True)
    (rank,n,i) = (1,1,0)
    while(i < len(scores)):
        j = i
        while((j < len(scores)-1) and (tmp[j][0] == tmp[j+1][0])):
            j += 1
        n = j-i+1
        for j in range(n):
            idx = tmp[i+j][1]
            ranks[idx] = rank
        rank += n
        i += n
    return ranks

def get_ranking(n, parameters):
    ranks = get_ranks(parameters)
    ranking = sorted(range(len(parameters)), key = lambda k: parameters[k])
    toppers = np.where(ranks == 1)[0]
    top = np.random.choice(toppers)
    return ranking, ranks, top
        
# RUNNER CODE
experiments = 10
iterations = 100
recovery_counts = np.zeros((n_items-1, experiments))
performance_factor = np.zeros((n_items-1, experiments))
L2_loss = np.zeros((n_items-1, experiments))
avg_time = np.zeros((n_items-1, experiments))
avg_count = np.zeros((n_items-1, experiments))

#Scores = np.delete(scores, 0)
Scores = np.copy(scores)
ssum = np.sum(Scores)
for i in range(0, n_items):
    Scores[i] = Scores[i]/ssum

for itr in range(experiments):
    for bgt in range(0, 5):
        Budget = (bgt + 2) * n_items
        #Budget = n_items * n_items
        rc = 0
        pf = 0
        ll = 0
        total_time = 0
        total_count = 0
        for run in range(iterations):
            start_time = time.time()
            
            count = 0
            data = []
            comp_matrix = np.zeros(((n_items+1), (n_items+1)))
            
            for i in range(1, n_items + 1):
                comp_matrix[i][i] = float("inf")
                comp_matrix[0][i] = float("inf")
                comp_matrix[i][0] = float("inf")
    
            A = np.arange(1, n_items+1)
            random.shuffle(A)
            while(count < Budget):
                temp = np.copy(A)
                #print("before", temp)
                quickSort(temp, 0, n_items-1)
                #print("after", temp)
                data.append(temp)
                
            cope = np.zeros(((n_items+1), (n_items+1)))
            cope[0][0] = float("inf")
            for i in range(1, n_items + 1):
                cope[i][i] = float("inf")
                cope[0][i] = float("inf")
                cope[i][0] = float("inf")
            
            for i in range(len(data)):
                for j in range(n_items):
                    for k in range(j + 1, n_items):
                        cope[data[i][k]][data[i][j]] += 1
            
            cope_scores = [0] * n_items
            for i in range(n_items):
                for j in range(i+1, n_items):
                    if(i != j):
                        if(cope[i+1][j+1] > cope[j+1][i+1]):
                            cope_scores[i] += 1
                        elif(cope[i+1][j+1] == cope[j+1][i+1]):
                            cope_scores[i] += 0.5
                            cope_scores[j] += 0.5
                        else:
                            cope_scores[j] += 1
            #ranking = sorted(range(len(cope_scores)), key = lambda k: cope_scores[k])
            
            #ranks = [0]*(len(ranking))
            #for j in range(0, len(ranking)):
            #    ranks[ranking[j]] = len(ranking) - j
            
            ranking, ranks, top = get_ranking(n_items, cope_scores)
            
            csum = np.sum(cope_scores)
            for i in range(0, n_items):
                cope_scores[i] = cope_scores[i]/csum
            
            end_time = time.time()
            ll += np.linalg.norm(Scores - cope_scores)
            if(true_top == top):
                rc += 1
            #pf += (abs(ranks[n_items-1]-1) + abs(ranks[n_items-2]-2) + abs(ranks[n_items-3]-3))/3
            pf += abs(ranks[true_top]-1)
            total_time += end_time - start_time
            total_count += count
        
        recovery_counts[bgt][itr] = rc
        performance_factor[bgt][itr] = pf/iterations
        L2_loss[bgt][itr] = ll/iterations
        avg_time[bgt][itr] = total_time/iterations
        avg_count[bgt][itr] = total_count/iterations

recovery_count_mean = np.mean(recovery_counts, axis=1)
recovery_count_var = np.var(recovery_counts, axis=1)

performance_factor_mean = np.mean(performance_factor, axis=1)
performance_factor_var = np.var(performance_factor, axis=1)

L2_loss_mean = np.mean(L2_loss, axis=1)
L2_loss_var = np.var(L2_loss, axis=1)

avg_time_mean = np.mean(avg_time, axis=1)
avg_time_var = np.var(avg_time, axis=1)

avg_count_mean = np.mean(avg_count, axis=1)
avg_count_var = np.var(avg_count, axis=1)
