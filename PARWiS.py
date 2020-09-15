"""
Install the choix library to run this code. Some functions in the code borrow implementation from the library.
https://github.com/lucasmaystre/choix
"""
import numpy as np
import choix
import random
import tqdm
from utils import init, get_args, print_metric
np.set_printoptions(precision=3, suppress=True)

def halve(m, a, scores, data, comp_matrix):
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
                data.append((p+1, q+1))
                comp_matrix[p+1][q+1] += 1
            else:
                win_q += 1
                data.append((q+1, p+1))
                comp_matrix[q+1][p+1] += 1
        num_votes += m
        if(win_p >= win_q):
            new_a[i] = p
        else:
            new_a[i] = q
    if(size % 2 == 1):
        new_a[int(size/2)] = a[size-1]
    return new_a, num_votes, data, comp_matrix

def SELECT(n, m, scores, data, comp_matrix):
    """
    Selects the topper among the given set of items using the SELECT algorithm routine.
    """
    a = np.arange(n)
    total_votes = 0
    while(np.size(a) > 1):
        a, num_votes, data, comp_matrix = halve(m, a, scores, data, comp_matrix)
        total_votes += num_votes
    top = a[0]
    return top, total_votes, data, comp_matrix

def reset(n, scores, initialization="heuristic"):
    """
    To intialize the algorithm using the first shoe-string of n-1 votes.

    initialization options:
    heuristic -> The default method as discussed in the paper.
    cyclic -> Compare (1,2), (2,3), ..., (n-1, n), (n, 1) to form a complete cycle of votes.
    SELECT -> Use the select algorithm routine to intialize the algorithm.
    """
    # Initializing the comparison matrix
    data = []
    comp_matrix = np.zeros((n+1, n+1))
    comp_matrix[0][0] = float("inf")
    for i in range(1, n+1):
        comp_matrix[i][i] = float("inf")
        comp_matrix[0][i] = float("inf")
        comp_matrix[i][0] = float("inf")
        data.append((i, 0))
        data.append((0, i))
    # Initializing the Shoe-String and getting the Ranking.
    if(initialization == "heuristic"):
        initial = n-1
        p = 1
        q = 2
        for i in range(initial):
            if(Vote(scores[p-1], scores[q-1])):
                data.append((p,q))
                comp_matrix[p][q] += 1
            else:
                data.append((q,p))
                comp_matrix[q][p] += 1
                p = q
            q = (q % n)+1
    elif(initialization == "cyclic"):
        initial = n
        p = 1
        q = 2
        for i in range(initial):
            if(Vote(scores[p-1], scores[q-1])):
                data.append((p,q))
                comp_matrix[p][q] += 1
            else:
                data.append((q,p))
                comp_matrix[q][p] += 1
            p = (p % n)+1
            q = (p % n)+1
    elif(initialization == "SELECT"):
        top, initial, data, comp_matrix = SELECT(n, 1, scores, data, comp_matrix)
    estimates = Rank_Centrality(n, data)
    estimates = normalize(estimates)
    # Initializing Markov Chain.
    chain = create_chain(n, data)
    A = np.identity(n+1, dtype = float) - chain
    A_hash = np.linalg.pinv(A)
    return data, initial, comp_matrix, chain, A_hash, estimates

def Vote(i, j):
    """
    Simulate voting on pair (a, b) when theit BTL scores are (i, j) respectively.
    """
    if(random.uniform(0, i+j) < i):
        return True
    else:
        return False

def normalize(array):
    """
    Normalize the BTL score vector such that sum of all scores is 1.
    """
    asum = np.sum(array)
    for i in range(0, len(array)):
        array[i] = array[i]/asum
    return array

def create_chain(n, data):
    """
    Create the underlying Markov Chain using the pairwise comparison data.
    """
    chain = np.zeros(((n+1), (n+1)))
    for winner, loser in data:
        chain[loser, winner] += 1.0
    # Transform the counts into ratios.
    idx = chain > 0  # Indices (i,j) of non-zero entries.
    chain[idx] = chain[idx] / (chain + chain.T)[idx]
    # Finalize the Markov chain by adding the self-transition rate.
    chain -= np.diag(chain.sum(axis=1))
    chain = chain / n
    for i in range(n+1):
        chain[i][i] += 1.0
    return chain

def get_ranks(scores):
    """
    Caluculate Ranks of items by assigning the average rank to the items with the same scores.
    """
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
            ranks[idx] = rank + (n-1)/2
        rank += n
        i += n
    return ranks

def get_ranking(n, parameters):
    """
    Get the current ranking of items, their ranks as well as the current topper.
    """
    params = np.delete(parameters, 0)
    ranks = get_ranks(params)
    ranking = sorted(range(len(parameters)), key = lambda k: parameters[k])
    ranking.remove(0)
    toppers = np.where(ranks == ranks.min())[0]
    top = np.random.choice(toppers) + 1
    return ranking, ranks, top

def IMC_update(n, comp_matrix, pair, inc, phi, chain, MC, est, metric="L2"):
    """
    Incremental Markov Chain update when one row of the chain has changed.
    There are various ways possible to caluculate the disruption score but we stick to the default L2-norm as discussed in the paper.
    """
    i = pair[0]
    j = pair[1]
    q = np.copy(chain[i])
    p = np.copy(q)
    if(inc):
        p[j] = ((comp_matrix[j][i])/(comp_matrix[j][i]+comp_matrix[i][j]+1))/n
    else:
        p[j] = ((comp_matrix[j][i]+1)/(comp_matrix[j][i]+comp_matrix[i][j]+1))/n
    p[i] = q[i] + q[j] - p[j]
    delta = q - p
    error = (phi[i]/(1 + np.dot(delta, MC[:, i]))) * np.dot(delta, MC)
    pi = phi - error
    gamma = np.dot(error, MC[:, i])/phi[i]
    temp = MC - (gamma * np.identity(n+1, dtype = float))
    IP_hash = MC + np.ones(n+1)[:, np.newaxis]*np.dot(error, temp) - (MC[:, i][:, np.newaxis]*error)/phi[i]
    pi = normalize(pi)
    new_chain = np.copy(chain)
    new_chain[i] = p
    ranking1, ranks1, top1 = get_ranking(n, pi)
    ranking2, ranks2, top2 = get_ranking(n, est)
    if(metric == "top_drop"):
        # Top Rank Drop
        return ranks1[top2-1]-ranks2[top2-1], pi, new_chain, IP_hash
    elif(metric == "L2"):
        # L2-norm of Difference in score vectors
        return np.linalg.norm(pi-est), pi, new_chain, IP_hash
    elif(metric == "L1"):
        # L1-norm of Difference in rank vectors
        return np.linalg.norm(ranks1 - ranks2, 1), pi, new_chain, IP_hash
    elif(metric == "abs_rank_diff"):
        # Absolute Rank Difference for the pair
        return (abs(ranks1[i-1]-ranks2[i-1])+abs(ranks1[j-1]-ranks2[j-1]))/2, pi, new_chain, IP_hash
    elif(metric == "drop_gain"):
        # Rank drop gain for the pair
        if(ranks2[i-1] > ranks2[j-1]):
            return ((ranks2[i-1]-ranks1[i-1])+(ranks1[j-1]-ranks2[j-1]))/2, pi, new_chain, IP_hash
        else:
            return ((ranks1[i-1]-ranks2[i-1])+(ranks2[j-1]-ranks1[j-1]))/2, pi, new_chain, IP_hash

def pick_a_pair(n, comp_matrix, estimates, top, chain, A_hash, ranking, k=1, method="weighted"):
    """
    Pick the next pair to be compared by evaluating the disruption score for the set of possible pairs.
    We only choose to compare the current topper with one of the rest to form a pair and reduce computation cost.
    """
    pair = np.random.choice(a = np.arange(1,n+1), size=2, replace=False)
    cand_pairs = []
    max_array = []
    for p in range(1,k+1):
        for q in range(p+1,n+1):
            i = ranking[n-p]
            j = ranking[n-q]
            # IMC for (i, j)
            m1, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (i, j), True, estimates, chain, A_hash, estimates)
            m1, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (j, i), False, pi, new_chain, IP_hash, estimates)
            # IMC for (j, i)
            m2, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (j, i), True, estimates, chain, A_hash, estimates)
            m2, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (i, j), False, pi, new_chain, IP_hash, estimates)
            # Update Max
            if(method == "average"):
                # Average
                m = m1+m2
            elif(method == "weighted"):
                # Weighted
                m = (estimates[i]*m1 + estimates[j]*m2)/(estimates[i]+estimates[j])
            cand_pairs.append((i,j))
            max_array.append(m)
    candidates = np.where(max_array == np.max(max_array))[0]
    idx = np.random.choice(candidates)
    pair = cand_pairs[idx]
    return pair

def random_pick(n, top):
    """
    To random pick one of the items apart from the topper.
    """
    item = random.randint(1,n-1)
    if(item >= top):
        item += 1
    return (top, item)

def Rank_Centrality(n, data):
    """
    Run the Rank Centrality algorithm to get a score vector based on the pairwise data seen so far.
    """
    params = choix.rank_centrality(n+1, data)
    estimates = np.exp(params)
    return estimates

def run_simulation_custom(n, toppers, experiments, iterations, budget, recovery_count, performance_factor):
    """
    Simulation for the case when the topper has a score x and rest have 100-x and we have an array of candidate x's.
    """
    aux_bgt = int(budget/int(budget/(n)))
    for tp, topper in tqdm.tqdm(enumerate(toppers), desc="toppers"):
        scores, true_top = init(n, topper=topper)
        for exp in tqdm.tqdm(range(experiments), desc="experiments"):
            for itr in tqdm.tqdm(range(iterations), desc="iterations"):
                data, initial, comp_matrix, chain, A_hash, estimates = reset(n, scores)
                ranking, ranks, top = get_ranking(n, estimates)
                for batch in range(int(budget/(n))):
                    for i in range(initial, aux_bgt):
                        (p,q) = pick_a_pair(n, comp_matrix, estimates, top, chain, A_hash, ranking)
                        if(Vote(scores[p-1], scores[q-1])):
                            data.append((p,q))
                            comp_matrix[p][q] += 1
                            m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (p,q), True, estimates, chain, A_hash, estimates)
                            m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (q,p), False, pi, new_chain, IP_hash, estimates)
                        else:
                            data.append((q,p))
                            comp_matrix[q][p] += 1
                            m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (q,p), True, estimates, chain, A_hash, estimates)
                            m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (p,q), False, pi, new_chain, IP_hash, estimates)
                        estimates = np.copy(pi)
                        ranking, ranks, top = get_ranking(n, estimates)
                        A_hash = np.copy(IP_hash)
                        chain = np.copy(new_chain)
                    ranking, ranks, top = get_ranking(n, estimates)
                    initial = 0
                    # Update the metrics
                    if(true_top == (top-1)):
                        recovery_count[tp][batch][exp] += 1
                    performance_factor[tp][batch][exp] += ranks[true_top]

    performance_factor /= iterations

    return ranking, ranks, data, scores, true_top, estimates, recovery_count, performance_factor

def run_simulation(n, experiments, iterations, budget, recovery_count, performance_factor, current_top,
                   precomputed=True, dataset=None):
    """
    Simulation of the algorithm for the synthetic and real-world dataset cases.
    """
    aux_bgt = int(budget/int(budget/(n)))
    scores, true_top = init(n, precomputed=precomputed, dataset=dataset)
    true_ranks = get_ranks(scores)
    for exp in tqdm.tqdm(range(experiments), desc="experiments"):
        for itr in tqdm.tqdm(range(iterations), desc="iterations"):
            data, initial, comp_matrix, chain, A_hash, estimates = reset(n, scores)
            ranking, ranks, top = get_ranking(n, estimates)
            for batch in range(int(budget/(n))):
                for i in range(initial, aux_bgt):
                    (p,q) = pick_a_pair(n, comp_matrix, estimates, top, chain, A_hash, ranking)
                    if(Vote(scores[p-1], scores[q-1])):
                        data.append((p,q))
                        comp_matrix[p][q] += 1
                        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (p,q), True, estimates, chain, A_hash, estimates)
                        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (q,p), False, pi, new_chain, IP_hash, estimates)
                    else:
                        data.append((q,p))
                        comp_matrix[q][p] += 1
                        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (q,p), True, estimates, chain, A_hash, estimates)
                        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (p,q), False, pi, new_chain, IP_hash, estimates)
                    estimates = np.copy(pi)
                    ranking, ranks, top = get_ranking(n, estimates)
                    A_hash = np.copy(IP_hash)
                    chain = np.copy(new_chain)
                ranking, ranks, top = get_ranking(n, estimates)
                initial = 0
                # Update the metrics
                if(true_top == (top-1)):
                    recovery_count[batch][exp] += 1
                performance_factor[batch][exp] += ranks[true_top]
                current_top[batch][exp] += true_ranks[top-1]

    performance_factor /= iterations
    current_top /= iterations

    return ranking, ranks, data, scores, true_top, estimates, recovery_count, performance_factor, current_top

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
        Budget = N*args.budget

        RC = np.zeros((int(Budget/(N)), Experiments))
        PF = np.zeros((int(Budget/(N)), Experiments))
        CT = np.zeros((int(Budget/(N)), Experiments))

        Ranking, Ranks, Data, Scores, True_top, Estimates, RC, PF, CT = run_simulation(N, Experiments, Iterations, Budget, RC, PF, CT,
                                                                                       precomputed=args.precomputed, dataset=args.dataset)

        print_metric("Recovery_Counts", RC)
        print_metric("Reported_Rank_of_True_Winner", PF)
        print_metric("True_Rank_of_Reported_Winner", CT)

    elif args.toppers is not None:
        Toppers = args.toppers

        N = args.n
        Experiments = args.experiments
        Iterations = args.iterations
        Budget = N*args.budget

        RC = np.zeros((len(Toppers), int(Budget/(N)), Experiments))
        PF = np.zeros((len(Toppers), int(Budget/(N)), Experiments))

        Ranking, Ranks, Data, Scores, True_top, Estimates, RC, PF = run_simulation_custom(N, Toppers, Experiments, Iterations, Budget, RC, PF)

        print_metric("Recovery_Counts", RC, Toppers)
        print_metric("Reported_Rank_of_True_Winner", PF, Toppers)

    else:
        N = args.n
        Experiments = args.experiments
        Iterations = args.iterations
        Budget = N*args.budget

        RC = np.zeros((int(Budget/(N)), Experiments))
        PF = np.zeros((int(Budget/(N)), Experiments))
        CT = np.zeros((int(Budget/(N)), Experiments))

        Ranking, Ranks, Data, Scores, True_top, Estimates, RC, PF, CT = run_simulation(N, Experiments, Iterations, Budget, RC, PF, CT,
                                                                                       precomputed=args.precomputed, dataset=args.dataset)

        print_metric("Recovery_Counts", RC)
        print_metric("Reported_Rank_of_True_Winner", PF)
        print_metric("True_Rank_of_Reported_Winner", CT)
