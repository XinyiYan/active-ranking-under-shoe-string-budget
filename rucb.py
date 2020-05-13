import numpy as np
import random, math, time
np.set_printoptions(precision=3, suppress=True)

'''
Class to implement Relative Upper Confidnence Bounds Algorithm.
'''
class rucb:
    def __init__(self, n, a, T):
        self.n = n
        self.a = a # read alpha
        self.T = T # budget
        self.W = np.zeros((n,n))
        self.B = {} # set B
        self.scores = np.zeros(n)
        #for i in range(n):
        #    self.scores[i] = random.uniform(0,75)
        #self.scores = np.sort(self.scores)
        #self.scores[n-1] = 100
        #self.true_top = n-1
        #self.scores = np.array([  8.2  , 100.   ,  48.174,  42.926,  52.671,  64.874,  27.607,
        #                        62.456,  64.792,  26.614])
        self.scores = np.array([ 2.637, 37.092, 35.992,  4.858, 23.636, 30.448, 45.242, 30.948,
                                29.136, 38.14 , 34.818, 29.57 , 30.447, 24.912, 28.451, 74.238,
                                73.627, 26.471, 35.294, 36.311, 70.924, 39.424, 64.263,  3.016, 40.923])
        #self.scores = np.array([69.72 , 66.96 , 59.993, 53.81 , 44.211, 43.364, 49.359, 12.563, 34.298, 49.05 , 14.32 , 45.721, 60.111, 37.027, 66.176, 47.118,
        #                        60.457, 38.228, 15.522, 65.783, 65.161, 39.038, 73.657, 52.103, 33.216, 65.346, 36.982, 44.408, 64.675, 40.606,  0.199, 42.142,
        #                        65.422, 69.91 , 29.524,  8.012, 17.244, 70.475, 57.783, 51.637, 46.264, 46.619,  8.721, 39.563, 53.239, 49.05 , 26.752, 37.529, 60.218, 44.554])
        self.true_top = np.argmax(self.scores)
        self.scores[self.true_top] = 100
        
    def compare(self, pair):
        i = pair[0]
        j = pair[1]
        a = self.scores[i]
        b = self.scores[j]
        if(random.uniform(0, a+b) < a):
            self.W[i][j] += 1
            return True
        else:
            self.W[j][i] += 1
            return False
        
    def get_top(self):
        for t in range(self.T):
            self.U = self.W/(self.W + self.W.transpose()) + np.sqrt(self.a*math.log(self.T)/(self.W + self.W.transpose()))
            np.place(self.U, np.isnan(self.U), 1)
            np.fill_diagonal(self.U, 0.5)
            
            self.C = {-1} # dumb but necessary
            for i in range(self.n):
                if(np.all(self.U[i] >= 0.5)):
                    self.C.add(i)
            if(len(self.C)==1):
                self.C.add(random.randint(0,n-1))
            self.C.remove(-1) # dumb but necessary
            if(len(self.B)==0):
                self.B = {-1}
            self.B.intersection_update(self.C)
            
            if(len(self.C) == 1):
                self.B = self.C
                for c in self.C:
                    break
            else:
                C_array = np.array(list(self.C))
                C_probabs = np.zeros(len(C_array))
                b_len = len(self.B)
                c_b_len = len(self.C.difference(self.B))
                for i in range(len(C_array)):
                    if(C_array[i] in self.B):
                        C_probabs = 0.5
                    else:
                        C_probabs = 1/((2**b_len)*c_b_len)
                c = np.random.choice(C_array, 1, C_probabs)[0]
            
            #d = np.argmax(self.U[:,c])
            #if(d == c):
            #    d = np.argsort(self.U[:,c])[-2]
            tmp = np.copy(self.U[:,c])
            tmp[c] = -np.inf
            d = np.random.choice(np.where(tmp == np.max(tmp))[0])
            
            self.compare((c,d))
        
        J = self.W/(self.W + self.W.transpose())
        np.place(J, np.isnan(J), 0.5)
        self.counts = np.zeros(self.n)
        for i in range(self.n):
            self.counts[i] = np.size(np.where(J[i] > 0.5))
        return np.argmax(self.counts)
    
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

'''
Runner Code.
'''
n = 25
experiments = 1
iterations = 1
recovery_counts = np.zeros((n-1, experiments))
performance_factor = np.zeros((n-1, experiments))
avg_time = np.zeros((n-1, experiments))

for itr in range(experiments):
    for bgt in range(0, 1):
        Initial = n
        Budget = (bgt+2)*n
        rc = 0
        pf = 0
        ll = 0
        total_time = 0
        for run in range(iterations):
            start_time = time.time()
            R = rucb(n, 0.51, Budget)
            top = R.get_top()
            scores = R.counts
            #ranking = sorted(range(len(scores)), key = lambda k: scores[k])
            #ranks = [0]*n
            #for i in range(n):
            #    ranks[ranking[i]] = n-i
            ranking, ranks, ttop = get_ranking(n, scores)
            end_time = time.time()
            if(ttop == R.true_top):
                rc += 1
            #pf += (abs(ranks[n-1]-1) + abs(ranks[n-2]-2) + abs(ranks[n-3]-3))/3
            pf += abs(ranks[R.true_top]-1)
            total_time += end_time - start_time
            print(itr, bgt, run)

        recovery_counts[bgt][itr] = rc
        performance_factor[bgt][itr] = pf/iterations
        avg_time[bgt][itr] = total_time/iterations

recovery_count_mean = np.mean(recovery_counts, axis=1)
recovery_count_var = np.var(recovery_counts, axis=1)

performance_factor_mean = np.mean(performance_factor, axis=1)
performance_factor_var = np.var(performance_factor, axis=1)

avg_time_mean = np.mean(avg_time, axis=1)
avg_time_var = np.var(avg_time, axis=1)