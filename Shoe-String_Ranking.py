from numba import jit, cuda
import numpy as np
import choix, random, math, time
np.set_printoptions(precision=3, suppress=True)

def init(n):
    scores = 75*np.random.rand(n)
    #for i in range(n):
    #    scores[i] = i+1
    #random.shuffle(scores)
    #scores = np.array([ 2.637, 37.092, 35.992,  4.858, 23.636, 30.448, 45.242, 30.948,
    #                   29.136, 38.14 , 34.818, 29.57 , 30.447, 24.912, 28.451, 74.238,
    #                   73.627, 26.471, 35.294, 36.311, 70.924, 39.424, 64.263,  3.016, 40.923])
    #scores = np.array([69.72 , 66.96 , 59.993, 53.81 , 44.211, 43.364, 49.359, 12.563, 34.298, 49.05 , 14.32 , 45.721, 60.111, 37.027, 66.176, 47.118,
    #                   60.457, 38.228, 15.522, 65.783, 65.161, 39.038, 73.657, 52.103, 33.216, 65.346, 36.982, 44.408, 64.675, 40.606,  0.199, 42.142,
    #                   65.422, 69.91 , 29.524,  8.012, 17.244, 70.475, 57.783, 51.637, 46.264, 46.619,  8.721, 39.563, 53.239, 49.05 , 26.752, 37.529, 60.218, 44.554])
    scores = np.array([  8.2  , 100.   ,  48.174,  42.926,  52.671,  64.874,  27.607, 62.456,  64.792,  26.614])
    true_top = np.argmax(scores)
    scores[true_top] = 100
    return scores, true_top

def reset(n, initial, scores):
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
    p = 1
    q = 2    
    for i in range(initial):
        if(Vote(scores[p-1], scores[q-1])):
            data.append((p,q))
            comp_matrix[p][q] += 1
        else:
            data.append((q,p))
            comp_matrix[q][p] += 1
            #'''
            p = q 
        q = (q % n)+1
        while(q == p):
            q = random.randint(1,n)
            #'''
        #p = (p % n)+1
        #q = (p % n)+1
    estimates = Rank_Centrality(n, data)
    estimates = normalize(estimates)
    # Initializing Markov Chain.
    chain = create_chain(n, data)
    A = np.identity(n+1, dtype = float) - chain
    A_hash = np.linalg.pinv(A)
    return data, comp_matrix, chain, A_hash, estimates

def Vote(i, j):
    if(random.uniform(0, i+j) < i):
        return True
    else:
        return False

def create_chain(n, data):
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

def normalize(array):
    asum = np.sum(array)
    for i in range(0, len(array)):
        array[i] = array[i]/asum
    return array

def IMC_update(n, comp_matrix, pair, inc, phi, chain, MC, est):
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
    # Best Metric - Top Rank Drop
    return ranks1[top2-1]-ranks2[top2-1], pi, new_chain, IP_hash
    # L2-norm of Difference in score vectors
    #return np.linalg.norm(pi-est), pi, new_chain, IP_hash
    # L1-norm of Difference in rank vectors
    #return np.linalg.norm(ranks1 - ranks2, 1), pi, new_chain, IP_hash
    # Absolute Rank Difference for the pair
    #return (abs(ranks1[i-1]-ranks2[i-1])+abs(ranks1[j-1]-ranks2[j-1]))/2, pi, new_chain, IP_hash
    # Rank drop gain for the pair
    #if(ranks2[i-1] > ranks2[j-1]):
    #    return ((ranks2[i-1]-ranks1[i-1])+(ranks1[j-1]-ranks2[j-1]))/2, pi, new_chain, IP_hash
    #else:
    #    return ((ranks1[i-1]-ranks2[i-1])+(ranks2[j-1]-ranks1[j-1]))/2, pi, new_chain, IP_hash

def pick_a_pair_orig(n, comp_matrix, estimates, top, chain, A_hash, ranking):
    pair = np.random.choice(a = np.arange(1,n+1), size=2, replace=False)
    compare = np.arange(1,n+1)
    compare = np.delete(compare, top-1)
    max_array = np.zeros(len(compare))
    for idx in range(len(compare)):
        i = compare[idx]
        j = top
        #IMC for (i, j)
        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (i, j), True, estimates, chain, A_hash, estimates)
        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (j, i), False, pi, new_chain, IP_hash, estimates)
        max_array[idx] = m
    candidates = np.where(max_array == np.max(max_array))[0]
    idx = np.random.choice(candidates)
    i = compare[idx]
    j = top
    pair = (i, j)
    return pair

def pick_a_pair(n, comp_matrix, estimates, top, chain, A_hash, ranking):
    pair = np.random.choice(a = np.arange(1,n+1), size=2, replace=False)
    cand_pairs = []
    max_array = []
    k = 1
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
            # Average
            #m = m1+m2
            # Weighted
            m = (estimates[i]*m1 + estimates[j]*m2)/(estimates[i]+estimates[j])
            cand_pairs.append((i,j))
            max_array.append(m)
    candidates = np.where(max_array == np.max(max_array))[0]
    idx = np.random.choice(candidates)
    pair = cand_pairs[idx]
    return pair

def random_pick(n, top):
    item = random.randint(1,n-1)
    if(item >= top):
        item += 1
    return (top, item)

def get_ranking(n, parameters):
    params = np.delete(parameters, 0)
    ranks = get_ranks(params)
    ranking = sorted(range(len(parameters)), key = lambda k: parameters[k])
    ranking.remove(0)
    toppers = np.where(ranks == 1)[0]
    top = np.random.choice(toppers) + 1
    return ranking, ranks, top

def Rank_Centrality(n, data):
    params = choix.rank_centrality(n+1, data)
    estimates = np.exp(params)
    return estimates

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

def Copeland_Step(n, est_agg):
    cope = np.zeros((n+1, n+1))
    cope[0][0] = float("inf")
    for i in range(1, n+1):
        cope[i][i] = float("inf")
        cope[0][i] = float("inf")
        cope[i][0] = float("inf")
    
    for i in range(len(est_agg)):
        for j in range(1, n+1):
            for k in range(j+1, n+1):
                if(abs(est_agg[i][j] - est_agg[i][k]) > 1e-9):
                    if(est_agg[i][j] > est_agg[i][k]):
                        cope[j][k] += 1
                    else:
                        cope[k][j] += 1
    
    cope_scores = np.zeros(n)
    for i in range(n):
        for j in range(i+1,n):
            if(cope[i+1][j+1] > cope[j+1][i+1]):
                cope_scores[i] += 1
            elif(cope[i+1][j+1] == cope[j+1][i+1]):
                cope_scores[i] += 0.5
                cope_scores[j] += 0.5
            else:
                cope_scores[j] += 1
    
    return cope_scores

def run_simulation_batch(n, experiments, iterations, budget, recovery_count, performance_factor, avg_time):
    initial = n
    aux_bgt = int(budget/int(budget/(2*n)))
    scores, true_top = init(n)
    for exp in range(experiments):
        for itr in range(iterations):
            #random.shuffle(scores)
            #true_top = np.argmax(scores)
            start_time = time.time()
            est_agg = []
            for batch in range(int(budget/(2*n))):
                data, comp_matrix, chain, A_hash, estimates = reset(n, initial, scores)
                ranking, ranks, top = get_ranking(n, estimates)
                for i in range(initial, aux_bgt):
                    (p,q) = pick_a_pair_orig(n, comp_matrix, estimates, top, chain, A_hash, ranking)
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
                est_agg.append(estimates)
                print(exp, itr, batch)
                
                cope_scores = Copeland_Step(n, est_agg)
                ranks = get_ranks(cope_scores)
            
                end_time = time.time()
                #if(top-1 == true_top):
                if(ranks[true_top] == 1):
                    recovery_count[batch][exp] += 1 
                performance_factor[batch][exp] += abs(ranks[true_top]-1)
                avg_time[batch][exp] += end_time - start_time
    
    performance_factor /= iterations
    avg_time /= iterations
    
    return ranks, est_agg, data, scores, true_top, cope_scores, recovery_count, performance_factor, avg_time

def run_simulation_full(n, experiments, iterations, budget, recovery_count, performance_factor, avg_time):
    initial = n
    aux_bgt = int(budget/int(budget/(2*n)))
    scores, true_top = init(n)
    for exp in range(experiments):
        for itr in range(iterations):
            start_time = time.time()
            data, comp_matrix, chain, A_hash, estimates = reset(n, initial, scores)
            ranking, ranks, top = get_ranking(n, estimates)
            for batch in range(int(budget/(2*n))):
                for i in range(initial, aux_bgt):
                    (p,q) = pick_a_pair_orig(n, comp_matrix, estimates, top, chain, A_hash, ranking)
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
                print(exp, itr, batch)
            
                end_time = time.time()
                if(ranks[true_top] == 1):
                    recovery_count[batch][exp] += 1 
                performance_factor[batch][exp] += abs(ranks[true_top]-1)
                avg_time[batch][exp] += end_time - start_time
    
    performance_factor /= iterations
    avg_time /= iterations
    
    return ranking, ranks, data, scores, true_top, estimates, recovery_count, performance_factor, avg_time

'''
RUNNER CODE
'''
N = 10
Experiments = 10
Iterations = 100
Budget = 10*N
RC = np.zeros((int(Budget/(2*N)), Experiments))
PF = np.zeros((int(Budget/(2*N)), Experiments))
AT = np.zeros((int(Budget/(2*N)), Experiments))

Ranks, Estimates_Agg, Data, Scores, True_top, Cope_Scores, RC, PF, AT = run_simulation_batch(N, Experiments, Iterations, Budget, RC, PF, AT)
#Ranking, Ranks, Data, Scores, True_top, Estimates, RC, PF, AT = run_simulation_full(N, Experiments, Iterations, Budget, RC, PF, AT)

recovery_count_mean = np.mean(RC, axis=1)
recovery_count_var = np.var(RC, axis=1)

performance_factor_mean = np.mean(PF, axis=1)
performance_factor_var = np.var(PF, axis=1)

avg_time_mean = np.mean(AT, axis=1)
avg_time_var = np.var(AT, axis=1)
