import numpy as np
import random
import tqdm
from utils import init, get_ranks, get_args, print_metric
np.set_printoptions(precision=3, suppress=True)

def Vote(i, j):
    """
    Simulate voting on pair (a, b) when theit BTL scores are (i, j) respectively.
    """
    if(random.uniform(0, i+j) < i):
        return True
    else:
        return False

def halve(m, a, scores):
    """
    Compares each item in the row in pairs of 2 for a times and prmotes the winner to the next row.
    """
    size = np.size(a)
    if(size % 2 == 0):
        new_a = np.arange(int(size/2))
    else:
        new_a = np.arange(int(size/2)+1)
    num_votes = 0
    random.shuffle(a)
    for i in range(int(size/2)):
        p = a[2*i]
        q = a[2*i+1]
        win_p = 0
        win_q = 0
        for j in range(m):
            if(Vote(scores[p], scores[q])):
                win_p += 1
            else:
                win_q += 1
        num_votes += m
        if(win_p >= win_q):
            new_a[i] = p
        else:
            new_a[i] = q
    if(size % 2 == 1):
        new_a[int(size/2)] = a[size-1]
    return new_a, num_votes

def SELECT(n, m, scores):
    """
    Selects the topper among the given set of items using the SELECT algorithm routine.
    """
    a = np.arange(n)
    total_votes = 0
    while(np.size(a) > 1):
        a, num_votes = halve(m, a, scores)
        total_votes += num_votes
    top = a[0]
    return top, total_votes

def run_simulation_custom(n, toppers, experiments, iterations, budget, recovery_count):
    """
    Simulation for the case when the topper has a score x and rest have 100-x and we have an array of candidate x's.
    """
    range_m = 1*np.arange(1,budget+1)
    for tp, topper in tqdm.tqdm(enumerate(toppers), desc="toppers"):
        scores, true_top = init(n, topper=topper)
        for exp in tqdm.tqdm(range(experiments), desc="experiments"):
            for itr in tqdm.tqdm(range(iterations), desc="iterations"):
                for b in range(budget):
                    m = range_m[b]
                    top, total_votes = SELECT(n, m, scores)
                    if(top == true_top):
                        recovery_count[tp][b][exp] += 1

    return scores, true_top, recovery_count

def run_simulation(n, experiments, iterations, budget, recovery_count, current_top,
                   precomputed=True, dataset=None):
    """
    Simulation of the algorithm for the synthetic and real-world dataset cases.
    """
    range_m = 1*np.arange(1,budget+1)
    scores, true_top = init(n, precomputed=precomputed, dataset=dataset)
    true_ranks = get_ranks(scores)
    for exp in tqdm.tqdm(range(experiments), desc="experiments"):
        for itr in tqdm.tqdm(range(iterations), desc="iterations"):
            for b in range(budget):
                m = range_m[b]
                top, total_votes = SELECT(n, m, scores)
                if(top == true_top):
                    recovery_count[b][exp] += 1
                current_top[b][exp] += true_ranks[top]

    current_top /= iterations

    return scores, true_top, recovery_count, current_top

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
        CT = np.zeros((Budget, Experiments))

        Scores, True_top, RC, CT = run_simulation(N, Experiments, Iterations, Budget, RC, CT,
                                                  precomputed=args.precomputed, dataset=args.dataset)

        print_metric("Recovery_Counts", RC)
        print_metric("True_Rank_of_Reported_Winner", CT)

    elif args.toppers is not None:
        Toppers = args.toppers

        N = args.n
        Experiments = args.experiments
        Iterations = args.iterations
        Budget = args.budget

        RC = np.zeros((len(Toppers), Budget, Experiments))

        Scores, True_top, RC = run_simulation_custom(N, Toppers, Experiments, Iterations, Budget, RC)

        print_metric("Recovery_Counts", RC, Toppers)

    else:
        N = args.n
        Experiments = args.experiments
        Iterations = args.iterations
        Budget = args.budget

        RC = np.zeros((Budget, Experiments))
        CT = np.zeros((Budget, Experiments))

        Scores, True_top, RC, CT = run_simulation(N, Experiments, Iterations, Budget, RC, CT,
                                                  precomputed=args.precomputed, dataset=args.dataset)

        print_metric("Recovery_Counts", RC)
        print_metric("True_Rank_of_Reported_Winner", CT)
