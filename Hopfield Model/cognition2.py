import math
import random

import numpy
import numpy as np
from matplotlib import pyplot as plt


def get_p_memory_patterns(f, n, p):
    memory_patterns = []
    for i in range(p):
        memory_patterns.append(np.array(random.choices([1, 0], [f, 1 - f], cum_weights=None, k=n)))
        # return a list (representing
        # a memory pattern) with length n (as the number of neurons in the network) with probability f that the i
        # coordinate will be 1
    return memory_patterns  # dimensions are PXN


def calculate_synaptic_strength(memory_patterns, n, f, p):
    memory_patterns_temp = np.array(memory_patterns)
    k = 1 / (f * (1 - f) * n)
    # calculate Jij
    f_matrix = np.full((p, n), f)
    memory_patterns_minus_f = numpy.subtract(memory_patterns_temp, f_matrix)
    memory_patterns_transposed_minus_f = numpy.subtract(np.transpose(memory_patterns_temp), np.transpose(f_matrix))
    connection_matrix = k * np.matmul(memory_patterns_transposed_minus_f, memory_patterns_minus_f)  # the ij coordinate
    # is for Jig
    # put zero in the coordinates in which i = j (on the diagonal)
    for i in range(n):
        connection_matrix[i][i] = 0
    return connection_matrix


""" to choose randomly p vectors in dimensions NX1 in module 2 field (with 0,1 values only) with f chances that the
coordinate 1 <= i <= n will be 1. all p vectors which represent memory pattern need to be different
and. to calculate the synaptic strength of synapse Jig (1<=i.j<=N, i=!j), put them in a matrix and to
 return this matrix from the function.
 @param n is the number of neurons in the network
 @param p is the number of memory patterns that the network need to learn
 @param f is the probability in which a neuron will be active in some memory pattern
 :return a connection matrix, which in the ij coordinate represent the strength of the synapse between the pre synaptic
 neuron j and the post synaptic neuron j.
 """


def question1(n, p, f):
    # to get p memory patterns following the instructions
    memory_patterns = get_p_memory_patterns(f, n, p)
    connection_matrix = calculate_synaptic_strength(memory_patterns, n, f, p)
    return connection_matrix


def heavyside(x):
    if x > 0:
        return 1
    else:
        return 0


""" this function calculate a dynamic of an a-synchrony Hopfield network
@param: connection_matrix is a matrix with the synaptic strength of the synapse connecting the pre-synaptic j neuron 
and the post-synaptic i neuron in the ij coordinate. this matrix is with dimensions NXN
@param: start_pattern_vector is a vector representing the start state of the neuronal activity of the network. this vector
is with dimensions NX1
@param: t is the threshold of the neurons 
:return the memory pattern that the network converged into
"""


def question2(connection_matrix, start_pattern_vector, t):
    current_pattern = None
    next_pattern = np.array(start_pattern_vector)
    n = next_pattern.size
    while not np.array_equal(current_pattern, next_pattern):
        current_pattern = np.array(next_pattern)
        for i in range(n):
            hi = np.dot(connection_matrix[i], next_pattern)
            next_pattern[i] = heavyside(hi - t)
            # next_pattern_vector[i] == si(t+1)
    return next_pattern


""" this function use question2 function and calculate the percentage (range [0,1]) of the neurons that there activity
 has change from the start pattern 
@param: connection_matrix is a matrix with the synaptic strength of the synapse connecting the pre-synaptic j neuron 
and the post-synaptic i neuron in the ij coordinate. this matrix is with dimensions NXN
@param: start_pattern_vector is a vector representing the start state of the neuronal activity of the network. this vector
is with dimensions NX1
@param: t is the threshold of the neurons 
return the percentage which have been calculated """


def question3(connection_matrix, start_pattern_vector, t):
    start_pattern = np.array(start_pattern_vector)
    n = start_pattern.size
    converged_pattern = question2(connection_matrix, start_pattern, t)
    sum_of_changes = 0
    for i in range(n):
        if start_pattern[i] != converged_pattern[i]:
            sum_of_changes += 1
    percentage = sum_of_changes / n
    return percentage


# create graph for question 4
def create_graph(f):
    plt.title(f"assignment 4, f = {f}")
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.xlabel("p/n")
    plt.ylabel("average mistake")

def get_list():
    list_a = []
    for alpha in range(2, 80, 4):
        list_a.append(alpha / 100)
    list_a.append(0.8)
    return list_a


def question4():
    for f in {0.1, 0.2, 0.3}:
        n = 1000
        a = 0.02
        create_graph(f)
        list_a = get_list()
        for a in list_a:
            mistake_percentage_sum_a = 0
            for i in range(5):
                p = math.ceil(a * n)
                memory_patterns = get_p_memory_patterns(f, n, p)
                connection_matrix = calculate_synaptic_strength(memory_patterns, n, f, p)
                mistake_percentage_sum_a += question3(connection_matrix, memory_patterns[0], 0.5 - f)
            average_mistake_a = mistake_percentage_sum_a / 5
            x = a
            y = average_mistake_a
            plt.plot(x, y, '.' + "b")

        plt.show()


question4()