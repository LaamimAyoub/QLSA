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
        base = list(range(1, nbrville + 1))
    else:
        base = list(range(nbrville))

    for i in range(n):
        temp = base[:]
        shuffle(temp)
        pop.append(temp)
    return pop

# cython: language_level=3
# distutils: extra_compile_args=-O3

import numpy as np
cimport numpy as np
import random
from libc.stdlib cimport rand, RAND_MAX

# Define the data type for the tour array elements
DTYPE = np.int32
ctypedef np.int32_t DTYPE_t

def double_bridge_kick_cy(np.ndarray[DTYPE_t, ndim=1] tour):
    """
    Performs a high-speed Double-Bridge Kick (4-change) perturbation
    on a TSP tour using Cython.

    The move rearranges the five segments S1, S2, S3, S4, S5 into
    the new order: S1, S4, S3, S2, S5.

    Args:
        tour (np.ndarray): The current sequence of cities (the tour).

    Returns:
        np.ndarray: The new tour after the double-bridge kick.
    """
    cdef Py_ssize_t n = tour.shape[0]
    cdef Py_ssize_t i1, i2, i3, i4
    cdef Py_ssize_t i
    cdef list cut_points_list

    if n < 8:
        # Not enough cities for a meaningful 4-change
        return tour

        # --- 1. Select 4 Distinct Cut Points ---
    # Use random.sample for ease of selecting distinct points.
    # Note: For maximum speed, this part could be done with C standard library
    # calls, but the Python overhead here is negligible compared to the C-speed
    # array manipulation below.
    cut_points_list = sorted(random.sample(range(1, n), 4))
    i1 = cut_points_list[0]
    i2 = cut_points_list[1]
    i3 = cut_points_list[2]
    i4 = cut_points_list[3]

    # --- 2. Create the New Tour Array ---
    # Initialize the new array with the same size and data type
    cdef np.ndarray[DTYPE_t, ndim=1] new_tour = np.empty_like(tour)
    cdef Py_ssize_t dest_idx = 0

    # --- 3. Copy Segments in New Order (S1, S4, S3, S2, S5) ---

    # S1: tour[0:i1] -> new_tour[0:i1]
    for i in range(i1):
        new_tour[dest_idx] = tour[i]
        dest_idx += 1

    # S4: tour[i3:i4] -> new_tour[dest_idx : dest_idx + (i4-i3)]
    for i in range(i3, i4):
        new_tour[dest_idx] = tour[i]
        dest_idx += 1

    # S3: tour[i2:i3] -> new_tour[dest_idx : dest_idx + (i3-i2)]
    for i in range(i2, i3):
        new_tour[dest_idx] = tour[i]
        dest_idx += 1

    # S2: tour[i1:i2] -> new_tour[dest_idx : dest_idx + (i2-i1)]
    for i in range(i1, i2):
        new_tour[dest_idx] = tour[i]
        dest_idx += 1

    # S5: tour[i4:n] -> new_tour[dest_idx : dest_idx + (n-i4)]
    for i in range(i4, n):
        new_tour[dest_idx] = tour[i]
        dest_idx += 1

    return new_tour

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

