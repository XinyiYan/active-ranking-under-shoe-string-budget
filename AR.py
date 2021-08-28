import numpy as np
import tqdm
from utils import init, get_ranks, get_ranking, get_args, print_metric
np.set_printoptions(precision=3, suppress=True)

############################

"""
Code has been borrowed from the official repository of the paper.
    Heckel, R.; Shah, N. B.; Ramchandran, K.; and Wainwright,
    M. J. 2019. "Active ranking from pairwise comparisons and
    when parametric assumptions do not help." The Annals of
    Statistics 47(6): 3099–3126.
https://github.com/reinhardh/supplement_active_ranking
"""

from numpy import *
import random
from itertools import permutations

'''
Model for pairwise comparisons
'''
class pairwise:
    def __init__(self,n,budget,precomputed=True,dataset=None):
        self.ctr = 0 # counts how many comparisons have been queried
        self.n = n
        self.budget = budget
        self.Scores, self.true_top = init(self.n, precomputed=precomputed, dataset=dataset)

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
                # print(self.pairwise.ctr)
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

def run_simulation(n, experiments, iterations, budget, recovery_count, performance_factor, current_top,
                   precomputed=True, dataset=None):
    """
    Simulation of the algorithm for the synthetic and real-world dataset cases.
    """
    for itr in tqdm.tqdm(range(experiments), desc="experiments"):
        for bgt in tqdm.tqdm(range(0, budget), desc="budget"):
            Budget = (bgt + 1) * n
            rc = 0
            pf = 0
            ct = 0
            total_time = 0
            total_count = 0
            for run in tqdm.tqdm(range(iterations), desc="iterations"):
                P = pairwise(n, Budget, precomputed=precomputed, dataset=dataset)
                T = topkalg(P, 1)
                T.rank()
                estimates = T.estimates

                ranking, ranks, top = get_ranking(n, estimates)
                true_ranks = get_ranks(P.Scores)

                if(P.true_top == top):
                    rc += 1
                pf += ranks[P.true_top]
                ct += true_ranks[top]

            recovery_count[bgt][itr] = rc
            performance_factor[bgt][itr] = pf/iterations
            current_top[bgt][itr] = ct/iterations
    return ranking, ranks, P.Scores, P.true_top, estimates, recovery_count, performance_factor, current_top

"""
Runner Code.
"""
if __name__ == "__main__":
    args = get_args()

    if args.dataset is not None:
        if args.dataset == "sushi-A":
            N = 10
        elif args.dataset == "sushi-B" or args.dataset == "jester" or args.dataset == "netflix" or args.dataset == "movielens":
            N = 100
        else:
            print("Invalid Dataset")
            exit()
        Experiments = args.experiments
        Iterations = args.iterations
        Budget = args.budget

        RC = np.zeros((Budget, Experiments))
        PF = np.zeros((Budget, Experiments))
        CT = np.zeros((Budget, Experiments))

        Ranking, Ranks, Scores, True_top, Estimates, RC, PF, CT = run_simulation(N, Experiments, Iterations, Budget, RC, PF, CT,
                                                                                 precomputed=args.precomputed, dataset=args.dataset)

        print_metric("Recovery_Counts", RC)
        print_metric("Reported_Rank_of_True_Winner", PF)
        print_metric("True_Rank_of_Reported_Winner", CT)

    else:
        N = args.n
        Experiments = args.experiments
        Iterations = args.iterations
        Budget = args.budget

        RC = np.zeros((Budget, Experiments))
        PF = np.zeros((Budget, Experiments))
        CT = np.zeros((Budget, Experiments))

        Ranking, Ranks, Scores, True_top, Estimates, RC, PF, CT = run_simulation(N, Experiments, Iterations, Budget, RC, PF, CT,
                                                                                 precomputed=args.precomputed, dataset=args.dataset)

        print_metric("Recovery_Counts", RC)
        print_metric("Reported_Rank_of_True_Winner", PF)
        print_metric("True_Rank_of_Reported_Winner", CT)
