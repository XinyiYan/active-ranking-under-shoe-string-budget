import numpy as np
import random, math
import tqdm
from utils import init, get_ranks, get_ranking, get_args, print_metric
np.set_printoptions(precision=3, suppress=True)

class rucb:
    """
    Class to implement Relative Upper Confidnence Bounds (RUCB) Algorithm.
    """
    def __init__(self, n, a, T, precomputed=True, dataset=None):
        self.n = n
        self.a = a # read alpha
        self.T = T # budget
        self.W = np.zeros((n,n))
        self.B = {} # set B
        self.scores, self.true_top = init(self.n, dataset=dataset, precomputed=precomputed)

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

def run_simulation(n, experiments, iterations, budget, recovery_count, performance_factor, current_top,
                   precomputed=True, dataset=None):
    """
    Simulation of the algorithm for the synthetic and real-world dataset cases.
    """
    for itr in tqdm.tqdm(range(experiments), desc="experiments"):
        for bgt in tqdm.tqdm(range(0, budget), desc="budget"):
            Initial = n
            Budget = (bgt+1)*n
            rc = 0
            pf = 0
            ct = 0
            for run in tqdm.tqdm(range(iterations), desc="iterations"):
                R = rucb(n, 0.51, Budget, precomputed=precomputed, dataset=dataset)
                top = R.get_top()
                scores = R.counts

                ranking, ranks, ttop = get_ranking(n, scores)
                true_ranks = get_ranks(R.scores)

                if(ttop == R.true_top):
                    rc += 1
                pf += ranks[R.true_top]
                ct += true_ranks[ttop]

            recovery_count[bgt][itr] = rc
            performance_factor[bgt][itr] = pf/iterations
            current_top[bgt][itr] = ct/iterations

    return ranking, ranks, R.scores, R.true_top, scores, recovery_count, performance_factor, current_top

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

        print(Scores)
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

        print(Scores)
        print_metric("Recovery_Counts", RC)
        print_metric("Reported_Rank_of_True_Winner", PF)
        print_metric("True_Rank_of_Reported_Winner", CT)
