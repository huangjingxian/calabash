def power_by_mtt(state, edges):
    """Calculate the total power of the state, by the matrix-tree theorem.
    """
    import numpy as np

    n = len(state)
    graph = np.zeros((n+1, n+1), dtype=np.float64)
    for u, v, w in edges:
        if (u == 0 or u in state) and v in state:
            graph[abs(u), abs(v)] = w
    mat_l = np.zeros((n+1, n+1), dtype=np.float64)
    for i in range(n+1):
        for j in range(n+1):
            if i == j:
                for k in range(n+1):
                    if k != i:
                        mat_l[i, j] += graph[k, i]
            else:
                mat_l[i, j] = -graph[i, j]
    det = np.linalg.det(mat_l[1:, 1:])
    return det


def randomized_algorithm():
    import pdb
    import random
    from math import log
    n, edges = read_input()
    best_all_state = None
    best_all_power = None
    # 925 2.53219314225e-27 938759288.271
    for seeds in range(2,100,2):
        # print seeds
        random.seed(seeds)
        times = 5000
        # best_state, best_power = None, None
        best_state = list(i * (-1)**random.randrange(1, 3) for i in range(1, n+1))
        best_power = power_by_mtt(tuple(best_state), edges)
        # print best_state
        for _ in range(times):
            pos = random.randrange(0, n)
            best_state[pos] *= (-1)
            rnd_power = power_by_mtt(tuple(best_state), edges)
            if best_power <= rnd_power:
                best_power = rnd_power
            if best_power > rnd_power:
                best_state[pos] *= (-1) 
            # print _, best_power
        if best_all_state ==  None or best_all_power<best_power:
            best_all_power = best_power
            best_all_state = best_state
    print ' '.join('%+d' % i for i in best_all_state)
    print seeds,best_all_power,1e6*log(best_all_power)+1e9


def read_input():
    def one_edge():
        line = raw_input()
        u, v, w = line.split()
        return int(u), int(v), float(w)
    n = int(raw_input())
    edges = [one_edge() for _ in range(4 * n**2 - 2*n)]
    return n, edges


if __name__ == '__main__':
    randomized_algorithm()