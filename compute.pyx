# compute.pyx
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3


import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time
from random import shuffle
from libc.math cimport exp

# calculate distante of a route
cpdef double compute_distance(list sol, object problem):
    cdef Py_ssize_t i, n = len(sol)
    cdef double s = 0
    for i in range(n - 1):
        #print(i)
        s += problem.get_weight(sol[i], sol[i + 1])
    s += problem.get_weight(sol[0], sol[n - 1])
    return s


# generate_tsp population
cpdef list generate_tsp(int n, int nbrville, bint has_node_coords):
    cdef list pop = []
    cdef list base
    cdef int i

    if has_node_coords:
        base = list(range(1, nbrville+1))
    else:
        base = list(range(nbrville))

    for i in range(n):
        temp = base[:]
        shuffle(temp)
        pop.append(temp)
    return pop



def softmax(np.ndarray[np.float64_t] q_values, double temperature):
    cdef int i, n = q_values.shape[0]
    cdef double max_q = np.max(q_values)
    cdef np.ndarray[np.float64_t] exp_q = np.empty(n)
    cdef double sum_exp = 0.0

    for i in range(n):
        exp_q[i] = exp((q_values[i] - max_q) / temperature)
        sum_exp += exp_q[i]

    return exp_q / sum_exp


cpdef int epsilon_greedy(np.ndarray[np.float64_t] q_values, double epsilon):
    cdef int n = q_values.shape[0]
    cdef double r = np.random.rand()
    cdef int best_idx = 0
    cdef int i

    if r < epsilon:
        return np.random.randint(0, n)
    else:
        for i in range(1, n):
            if q_values[i] > q_values[best_idx]:
                best_idx = i
        return best_idx

