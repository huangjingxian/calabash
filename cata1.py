import pdb
import random
from math import log
import numpy as np

def origin_graph(n,edges):
    tot_graph = np.zeros((2*n+1, 2*n+1), dtype=np.float64)
    for u, v, w in edges:
        posu,posv=0,0
        if u<0:
            posu=n+abs(u)
        else:
            posu = u
        if v<0:
            posv = n+abs(v)
        else:
            posv = v
        tot_graph[posu,posv]=w
    return tot_graph


def power_by_mtt(state, edges,n,graph):
    """Calculate the total power of the state, by the matrix-tree theorem.
    """
    
    mat_l = -np.copy(graph)
    for i in range(n+1):
        sums = np.sum(graph[:,i])-graph[i,i]
        mat_l[i,i]=sums
    det = np.linalg.det(mat_l[1:, 1:])
    return det

def new_graph(state,graph,pos):
    if state[pos] < 0:
        pospos = n+abs(state[pos])
    else:
        pospos = state[pos]
    for i in range(len(state)):
        if state[i] < 0:
            posi = n+abs(state[i])
        else:
            posi = state[i]
        w1 = tot_graph[pospos,posi]
        w2 = tot_graph[posi,pospos]
        graph[abs(state[pos]),abs(state[i])]=w1
        graph[abs(state[i]),abs(state[pos])]=w2
    graph[0,abs(state[pos])] = tot_graph[0,pospos] 

    return graph



def randomized_algorithm():


    best_all_state = None
    best_all_power = None
    best_seeds = -1

    for seeds in range(1500,3000,2):
        random.seed(seeds)
        times = 1500
        best_state = list(i * (-1)**random.randrange(1, 3) for i in range(1, n+1))
        # print best_state
        graph = np.zeros((n+1, n+1), dtype=np.float64)
        for u, v, w in edges:
            if (u == 0 or u in best_state) and v in best_state:
                graph[abs(u), abs(v)] = w
        # print "lll"
        best_power = power_by_mtt(best_state,edges,n,graph)
        # print "hhh"
        for _ in range(times):
            pos = random.randrange(0, n)
            best_state[pos] *= (-1)
            cp_graph = np.copy(graph)
            graph = new_graph(best_state,graph, pos)
            rnd_power = power_by_mtt(best_state, edges,n,graph)
            if best_power <= rnd_power:
                best_power = rnd_power
            if best_power > rnd_power:
                best_state[pos] *= (-1)
                graph = cp_graph 
            # print _, best_power
        if best_all_state ==  None or best_all_power<best_power:
            best_all_power = best_power
            best_all_state = best_state
            best_seeds = seeds
    print ' '.join('%+d' % i for i in best_all_state)
    print best_seeds,best_all_power,
    if best_all_power > 0:
        print 1e6*log(best_all_power)+1e9


def read_input():
    def one_edge():
        line = raw_input()
        u, v, w = line.split()
        return int(u), int(v), float(w)
    n = int(raw_input())
    edges = [one_edge() for _ in range(4 * n**2 - 2*n)]
    return n, edges


if __name__ == '__main__':
    n,edges = read_input()
    tot_graph = origin_graph(n,edges)
    # print tot_graph[501,2]
    randomized_algorithm()
