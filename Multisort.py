import numpy as np
import random
import tqdm
from utils import init, get_ranks, get_ranking, get_args, print_metric
np.set_printoptions(precision=3, suppress=True)

def compare(a, b, scores, count, Budget):
    """
    Compares the pair (a, b) with BTL scores if there is budget left else performs a random vote.
    """
    if(count < Budget):
        if(random.uniform(0, scores[a-1]+scores[b-1]) < scores[a-1]):
            return False
        else:
            return True
    else:
        if(random.uniform(0, 1) < 0.5):
            return False
        else:
            return True

def partition(arr, low, high, scores, count, Budget):
    """
    Partition the array with a randomly selected pivot.
    """
    i = (low - 1)
    pivot = arr[random.randint(low, high)]

    for j in range(low , high):

        if compare(arr[j], pivot, scores, count, Budget):
            count += 1
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i+1], arr[high] = arr[high], arr[i+1]
    return (i + 1), count

def quickSort(arr, low, high, scores, count, Budget):
    """
    The recursive quicksort algorithm which is used multiple times.
    """
    if low < high:

        pi, count = partition(arr, low, high, scores, count, Budget)

        count = quickSort(arr, low, pi-1, scores, count, Budget)
        count = quickSort(arr, pi+1, high, scores, count, Budget)
    return count

def Copeland_Step(n, data):
    """
    Copeland Aggregation over the obtained rankings.
    """
    cope = np.zeros(((n+1), (n+1)))
    cope[0][0] = float("inf")
    for i in range(1, n + 1):
        cope[i][i] = float("inf")
        cope[0][i] = float("inf")
        cope[i][0] = float("inf")

    for i in range(len(data)):
        for j in range(n):
            for k in range(j + 1, n):
                cope[data[i][k]][data[i][j]] += 1

    cope_scores = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if(i != j):
                if(cope[i+1][j+1] > cope[j+1][i+1]):
                    cope_scores[i] += 1
                elif(cope[i+1][j+1] == cope[j+1][i+1]):
                    cope_scores[i] += 0.5
                    cope_scores[j] += 0.5
                else:
                    cope_scores[j] += 1

    return cope_scores

def run_simulation(n, experiments, iterations, budget, recovery_count, performance_factor, current_top,
                   precomputed=True, dataset=None):
    """
    Simulation of the algorithm for the synthetic and real-world dataset cases.
    """
    scores, true_top = init(n, precomputed=precomputed, dataset=dataset)
    true_ranks = get_ranks(scores)

    for itr in tqdm.tqdm(range(experiments), desc="experiments"):
        for bgt in tqdm.tqdm(range(0, budget), desc="budget"):
            Budget = (bgt + 1) * n
            rc = 0
            pf = 0
            ct = 0
            for run in tqdm.tqdm(range(iterations), desc="iterations"):
                count = 0
                data = []

                A = np.arange(1, n+1)
                random.shuffle(A)
                while(count < Budget):
                    temp = np.copy(A)
                    count = quickSort(temp, 0, n-1, scores, count, Budget)
                    data.append(temp)

                cope_scores = Copeland_Step(n, data)

                ranking, ranks, top = get_ranking(n, cope_scores)

                csum = np.sum(cope_scores)
                for i in range(0, n):
                    cope_scores[i] = cope_scores[i]/csum

                if(true_top == top):
                    rc += 1
                pf += ranks[true_top]
                ct += true_ranks[top]

            recovery_count[bgt][itr] = rc
            performance_factor[bgt][itr] = pf/iterations
            current_top[bgt][itr] = ct/iterations
    return ranking, ranks, data, scores, true_top, cope_scores, recovery_count, performance_factor, current_top

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

        Ranking, Ranks, Data, Scores, True_top, Estimates, RC, PF, CT = run_simulation(N, Experiments, Iterations, Budget, RC, PF, CT,
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

        Ranking, Ranks, Data, Scores, True_top, Estimates, RC, PF, CT = run_simulation(N, Experiments, Iterations, Budget, RC, PF, CT,
                                                                                       precomputed=args.precomputed, dataset=args.dataset)

        print_metric("Recovery_Counts", RC)
        print_metric("Reported_Rank_of_True_Winner", PF)
        print_metric("True_Rank_of_Reported_Winner", CT)
