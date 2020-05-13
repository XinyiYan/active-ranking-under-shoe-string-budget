from numpy import *
import random
import time
from itertools import permutations
    
'''
Model for pairwise comparisons
'''
class pairwise:
    def __init__(self,n,budget):
        self.ctr = 0 # counts how many comparisons have been queried
        self.n = n
        self.budget = budget
        self.Scores = zeros(n)
        #for i in range(n):
        #    self.Scores[i] = random.uniform(0,75) # n-i
        #self.Scores = sort(self.Scores)
        #self.Scores[0] = 100
        #self.Scores = array([  8.2  , 100.   ,  48.174,  42.926,  52.671,  64.874,  27.607,
        #                     62.456,  64.792,  26.614])
        self.Scores = array([ 2.637, 37.092, 35.992,  4.858, 23.636, 30.448, 45.242, 30.948,
                             29.136, 38.14 , 34.818, 29.57 , 30.447, 24.912, 28.451, 74.238,
                             73.627, 26.471, 35.294, 36.311, 70.924, 39.424, 64.263,  3.016, 40.923])
        #self.Scores = array([69.72 , 66.96 , 59.993, 53.81 , 44.211, 43.364, 49.359, 12.563, 34.298, 49.05 , 14.32 , 45.721, 60.111, 37.027, 66.176, 47.118,
        #                     60.457, 38.228, 15.522, 65.783, 65.161, 39.038, 73.657, 52.103, 33.216, 65.346, 36.982, 44.408, 64.675, 40.606,  0.199, 42.142,
        #                     65.422, 69.91 , 29.524,  8.012, 17.244, 70.475, 57.783, 51.637, 46.264, 46.619,  8.721, 39.563, 53.239, 49.05 , 26.752, 37.529, 60.218, 44.554])
        self.true_top = argmax(self.Scores)
        self.Scores[self.true_top] = 100
         
    def random_uniform(self): 
        '''
        generate random pairwise comparison mtx with entries uniform in [0,1]
        '''
        self.P = random.rand(self.n,self.n)*0.9
        for i in range(n):
            self.P[i,i] = 0.5
        for i in range(n):
            for j in range(i+1,n):
                self.P[i,j] = 1 - self.P[j,i]
        self.sortP()

    def sortP(self):
        # sort the matrix according to scores
        scores = self.scores()
        pi = argsort(-scores)
        self.P = self.P[:,pi]
        self.P = self.P[pi,:]
    
    def generate_BTL(self,sdev=1):
        self.P = zeros((self.n,self.n))
        # Gaussian seems reasonable; 
        # if we choose it more extreme, e.g., like Gaussian^2 it looks
        # very different than the real-world distributions
        w = sdev*random.randn(self.n)
        self.w = w
        # w = w - min(w) does not matter
        for i in range(self.n):
            for j in range(i,self.n):
                self.P[i,j] = 1/( 1 + exp( w[j] - w[i] ) )
                self.P[j,i] = 1 - self.P[i,j]
        self.sortP()

    def uniform_perturb(self,sdev=0.01):
        for i in range(self.n):
            for j in range(i,self.n):
                perturbed_entry = self.P[i,j] + sdev*(random.rand()-0.5)
                if perturbed_entry > 0 and perturbed_entry < 1:
                    self.P[i,j] = perturbed_entry
                    self.P[j,i] = 1-perturbed_entry

    def generate_deterministic_BTL(self,w):
        self.w = w
        self.P = zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(i,self.n):
                self.P[i,j] = 1/( 1 + exp( w[j] - w[i] ) )
                self.P[j,i] = 1 - self.P[i,j]
        self.sortP()
    
    def generate_const(self,pmin = 0.25):
        self.P = zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(i+1,self.n):
                self.P[i,j] = 1 - pmin
                self.P[j,i] = pmin
    
    def compare(self,i,j):
        if(self.ctr < self.budget):
            #print("alright", self.ctr)
            if(random.uniform(0,self.Scores[i]+self.Scores[j]) < self.Scores[i]):
                return 1
            else:
                return 0
        else:
            #print("rand")
            if(random.uniform(0,1) < 0.5):
                return 1
            else:
                return 0
        #self.ctr += 1
        """
        if random.random() < self.P[i,j]:
            return 1 # i beats j
        else:
            return 0 # j beats i
        """

    def scores(self):
        P = array(self.P)
        for i in range(len(P)):
            P[i,i] = 0
        return sum(P,axis=1)/(self.n-1)

    def plot_scores(self):
        plt.plot(range(self.n), self.scores(), 'ro')
        plt.show()

    def top1H(self):
        sc = self.scores();
        return 1/(sc[0]-sc[1])**2 + sum([ 1/(sc[0]-sc[1])**2 for i in range(1,self.n)])

    def top1parH(self):
        sc = self.scores();
        w = sorted(self.w,reverse=True)
        return (( exp(w[0])-exp(w[1]) )/( exp(w[0])+exp(w[1]) ))**-2 + sum([ (( exp(w[0])-exp(w[i]) )/( exp(w[0])+exp(w[i]) ))**-2 for i in range(1,self.n)])

'''
Top k ranking algorithm: the active ranking algorithm tailored to top-k identification
Input:
- pairwise: A class abstracting a pairwise comparison model (see pairwise.py). 
The algorithm interacts with the model through asking for a comparison between 
item i and j by calling pairwise.compare(i,j)
- k: The number of top items to identify
- rule: different choices for confidence intervals, the default one is the one from the paper
'''
class topkalg:
    def __init__(self,pairwise,k,default_rule = None,epsilon=None):
        self.pairwise = pairwise # instance of pairwise
        self.k = k
        self.estimates = zeros(self.pairwise.n)
        if epsilon == None:
            self.epsilon = 0
        else:
            self.epsilon = epsilon
        
        if default_rule == None:
            self.default_rule = 0
        else:
            self.default_rule = default_rule


    def rank(self,delta=0.1,rule=None):
        if rule == None:
            rule = self.default_rule
            #print( "Use default rule: ", rule )

        self.pairwise.ctr = 0
        self.topitems = []        # estimate of top items
        self.ranking = []
        self.ranks = []
        # active set contains pairs (index, score estimate)
        active_set = [(i,0.0) for i in range(self.pairwise.n)]
        k = self.k    
        t = 1 # algorithm time
        while len(active_set) - k > 0 and k > 0:
            if rule == 0:
                alpha = sqrt( log( 3*self.pairwise.n*log(1.12*t)/delta ) / t ) # 5
            if rule == 1:
                alpha = sqrt( 2*log( 1*(log(t)+1) /delta) / t )
            if rule == 2: # this is the choice in Urvoy 13, see page 3
                alpha = 2*sqrt( 1/(2.0*t) * log(3.3*self.pairwise.n*t**2/delta) )
            if rule == 3:
                alpha = sqrt( 1.0/t * log(self.pairwise.n*log(t+2)/delta) )
            if rule == 4:
                alpha = sqrt( log(self.pairwise.n/3*(log(t)+1) /delta) / t )
            if rule == 5:
                alpha = 4*sqrt( 0.75 * log( self.pairwise.n * (1+log(t)) / delta ) / t )
            if rule == 6:
                alpha = 2*sqrt( 0.75 * log( self.pairwise.n * (1+log(t)) / delta ) / t )
            # for top-2 identification we can use a factor 2 instead of 4 from the paper, and the same guarantees hold
            if rule == 7:
                alpha = 2*sqrt( 0.5 * (log(self.pairwise.n/delta) + 0.75*log(log(self.pairwise.n/delta)) + 1.5*log(1 + log(0.5*t))) / t )

            ## update all scores
            for ind, (i,score) in enumerate(active_set):
                j = random.choice(range(self.pairwise.n-1))
                if j >= i:
                    j += 1
                xi = self.pairwise.compare(i,j)    # compare i to random other item
                self.pairwise.ctr += 1
                active_set[ind] = (i, (score*(t-1) + xi)/t)
                self.estimates[active_set[ind][0]] = active_set[ind][1]
            ## eliminate variables
            # sort descending by score
            active_set = sorted(active_set, key=lambda ind_score: ind_score[1],reverse=True)
            toremove = []
            totop = 0
            # remove top items
            for ind,(i,score) in enumerate(active_set):
                if(score - active_set[k][1] > alpha - self.epsilon):
                    self.topitems.append(i)
                    toremove.append(ind)
                    totop += 1
                else:
                    break # for all coming ones, the if condition can't be satisfied either
            # remove bottom items
            for ind,(i,score) in reversed(list(enumerate(active_set))):
                if(active_set[k-1][1] - score  > alpha - self.epsilon ):
                    toremove.append(ind)
                else:
                    break # for all coming ones, the if condition can't be satisfied either
            toremove.sort()    
            for ind in reversed(toremove):
                self.estimates[active_set[ind][0]] = -1
                del active_set[ind]
            k = k - totop
            t += 1
            
            if(self.pairwise.ctr >= self.pairwise.budget):
                #print("breaking")
                break
                
                
    def evaluate_perfect_recovery(self):
        origsets = []
        return (set(self.topitems) == set(range(self.k)))


############################

def get_ranks(scores):
    ranks = zeros(len(scores))
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
    toppers = where(ranks == 1)[0]
    top = random.choice(toppers)
    return ranking, ranks, top

# RUNNER CODE
global n_items
n_items = 25
experiments = 10
iterations = 100
recovery_counts = zeros((n_items-1, experiments))
avg_time = zeros((n_items-1, experiments))
avg_count = zeros((n_items-1, experiments))
performance_factor = zeros((n_items-1, experiments))

for itr in range(experiments):
    for bgt in range(0, 1):
        Budget = (bgt + 2) * n_items
        #Budget = n_items * n_items
        rc = 0
        pf = 0
        total_time = 0
        total_count = 0
        for run in range(iterations):
            start_time = time.time()
            
            P = pairwise(n_items, Budget)
            T = topkalg(P, 1)
            T.rank()
            estimates = T.estimates
            #ranking = sorted(range(len(estimates)), key = lambda k: estimates[k])
            #ranks = [0]*(len(ranking))
            #for j in range(0, len(ranking)):
            #    ranks[ranking[j]] = len(ranking) - j
            ranking, ranks, top = get_ranking(n_items, estimates)
            if(P.true_top == top):
                rc += 1
            
            end_time = time.time()
            pf += abs(ranks[P.true_top]-1)
            total_time += end_time - start_time
            total_count += T.pairwise.ctr
            #print(bgt, run)
        
        recovery_counts[bgt][itr] = rc
        avg_time[bgt][itr] = total_time/iterations
        avg_count[bgt][itr] = total_count/iterations
        performance_factor[bgt][itr] = pf/iterations

recovery_count_mean = mean(recovery_counts, axis=1)
recovery_count_var = var(recovery_counts, axis=1)

avg_time_mean = mean(avg_time, axis=1)
avg_time_var = var(avg_time, axis=1)

avg_count_mean = mean(avg_count, axis=1)
avg_count_var = var(avg_count, axis=1)

performance_factor_mean = mean(performance_factor, axis=1)
performance_factor_var = var(performance_factor, axis=1)
