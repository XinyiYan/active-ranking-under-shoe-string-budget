import numpy as np
import random
import tqdm
from utils import init, get_ranks, get_ranking, get_args, print_metric
np.set_printoptions(precision=3, suppress=True)

def Vote(i, j, budget, count):
    """
    Simulate voting on pair (a, b) when theit BTL scores are (i, j) respectively if the budget hasn't expired else random comparison.
    """
    if(count < budget):
        if(random.uniform(0, i+j) < i):
            return True
        else:
            return False
    else:
        if(random.uniform(0, 1) < 0.5):
            return True
        else:
            return False

def eliminate(n, t, T, k, A, P0, S0, delta):
    """
    Elimination step of the SECS Algorithm.
    """
    P = P0*(n/t)
    S = S0*(n/((n-1)*t))
    th = 2*np.log10(4*(n**2)*(t**2)/delta)/(t/n)
    C_t = (np.sqrt(th) + th/3)*(k+1)*0.5
    th = (n/(n-1))*np.sqrt(2*np.log10(4*n*(t**2)/delta)/t)
    l = np.size(A)
    new_A = np.copy(A)
    for x in range(l):
        for y in range(x+1, l):
            i = A[x]
            j = A[y]
            if(S[i] > (S[j] + th)):
                new_A = new_A[new_A != j]
            if(S[j] > (S[i] + th)):
                new_A = new_A[new_A != i]
            if(t > T):
                omega = np.array([])
                for w in range(n):
                    if((not w == i) and (not w == j)):
                        omega = np.append(omega, abs(P[i][w] - P[j][w]))
                    else:
                        omega = np.append(omega, 0)
                o_set = np.argsort(omega)[::-1]
                delta_ij = 2*(P[i][j] - 0.5)
                delta_ji = 2*(P[j][i] - 0.5)
                for w in range(k):
                    delta_ij += (P[i][w] - P[j][w])
                    delta_ji += (P[j][w] - P[i][w])
                if(delta_ij > C_t):
                    new_A = new_A[new_A != j]
                if(delta_ji > C_t):
                    new_A = new_A[new_A != i]
    return new_A, P0, S0

def sparse_borda(n, t, T, k, A, P, S, delta, scores, budget, count, r_count, data):
    """
    Successive Elimination with Comparison Sparsity (SECS) Algorithm.
    """
    if(r_count > 0):
        count -= r_count
        t -= 1
        while(r_count > 0):
            idx = len(data) - 1
            if(data[idx][0] != -1 and data[idx][1] != -1):
                P[data[idx][0]][data[idx][1]] -= 1
                S[data[idx][0]] -= 1
            data.pop(idx)
            r_count -= 1
    while(np.size(A) > 1 and ((budget - count) > np.size(A)/2)): # count < budget
        i = np.random.choice(n, 1)[0]
        l = np.size(A)
        r_count = 0
        for x in range(l):
            j = A[x]
            if(j != i):
                if(not Vote(scores[i], scores[j], budget, count)):
                    P[j][i] += 1
                    S[j] += 1
                    data.append((j, i))
                else:
                    data.append((-1, -1))
                count += 1
                r_count += 1
        A, P, S = eliminate(n, t, T, k, A, P, S, delta)
        t += 1
    est = np.zeros(n)
    S0 = S*(n/((n-1)*t))
    est[A] = S0[A]
    if(budget >= count):
        r_count = 0
    return est, t, A, P, S, count, r_count, data

def run_simulation(n, experiments, iterations, budget, recovery_count, performance_factor, current_top,
                   precomputed=True, dataset=None):
    """
    Simulation of the algorithm for the synthetic and real-world dataset cases.
    """
    T = 0
    k = 5
    delta = 0.1
    range_m = 1*np.arange(1, budget+1)
    scores, true_top = init(n, precomputed=precomputed, dataset=dataset)
    true_ranks = get_ranks(scores)

    for exp in tqdm.tqdm(range(experiments), desc="experiments"):
        for itr in tqdm.tqdm(range(iterations), desc="iterations"):
            t = 1
            count = 0
            A = np.arange(n)
            P = np.zeros((n, n))
            S = np.zeros(n)
            r_count = 0
            data = []
            for b in range(budget):
                m = range_m[b]

                est, t, A, P, S, count, r_count, data = sparse_borda(n, t, T, k, A, P, S, delta, scores, m*n, count, r_count, data)

                ranking, ranks, top = get_ranking(n, est)
                if(top == true_top):
                    recovery_count[b][exp] += 1
                performance_factor[b][exp] += ranks[true_top]
                current_top[b][exp] += true_ranks[top]

    current_top /= iterations
    performance_factor /= iterations

    return ranking, ranks, data, scores, true_top, est, recovery_count, performance_factor, current_top

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

        Ranking, Ranks, Data, Scores, True_top, Estimates, RC, PF, CT = run_simulation(N, Experiments, Iterations, Budget, RC, PF, CT,
                                                                                       precomputed=args.precomputed, dataset=args.dataset)

        print(Scores)
        print_metric("Recovery_Counts", RC)
        print_metric("Reported_Rank_of_True_Winner", PF)
        print_metric("True_Rank_of_Reported_Winner", CT)
