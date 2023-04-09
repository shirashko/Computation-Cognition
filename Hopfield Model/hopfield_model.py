import math
import random
import numpy as np
from matplotlib import pyplot as plt


# Background:
#
# The Hopfield model is a model of associative memory. It describes how a particular selection of synaptic connections
# in the network will cause the network to boot in a state "close" to a particular memory, and then converge to the
# memory itself after some dynamics.
#
# The model examines two important factors:
#   - How "close" the network needs to be rebooted in order for it to converge into the right memory.
#   - How many memories can be embedded in such a network.

def _get_p_memory_patterns(f, n, p):
    """
    this function calculate a memory patterns matrix size pXn, of p memory patterns, each memory pattern represent the
    activity of neurons in the network, when the probability a neuron in the network will be active in the pattern is f
    1 for active neuron, 0 otherwise

    Returns:
        the memory patterns matrix
    """
    memory_patterns = []
    for i in range(p):
        memory_patterns.append(np.array(random.choices([1, 0], [f, 1 - f], cum_weights=None, k=n)))
        # return a list (representing a memory pattern) with length n (as the number of neurons in the network) with
        # probability f that the i coordinate will be 1
    return memory_patterns  # dimensions are PXN


def calculate_synaptic_strength(memory_patterns, n, f, p):
    memory_patterns_temp = np.array(memory_patterns)
    k = 1 / (f * (1 - f) * n)
    # calculate Jij
    f_matrix = np.full((p, n), f)
    memory_patterns_minus_f = np.subtract(memory_patterns_temp, f_matrix)
    memory_patterns_transposed_minus_f = np.subtract(np.transpose(memory_patterns_temp), np.transpose(f_matrix))
    connection_matrix = k * np.matmul(memory_patterns_transposed_minus_f, memory_patterns_minus_f)  # the ij coordinate
    # is for Jig
    np.fill_diagonal(connection_matrix, 0)  # put zero in the coordinates in which i = j, because there isn't a synapse
    # between a neuron and itself
    return connection_matrix


def find_connection_matrix(n, p, f):
    """
    Randomly Choose p vectors in dimensions NX1 in module 2 field (with 0,1 values only) with f chances that the
    coordinate 1 <= i <= n will be 1. all p vectors which represent memory pattern need to be different
    and. to calculate the synaptic strength of synapse Jig (1<=i.j<=N, i=!j), put them in a matrix and to
    return this matrix from the function.

    Args:
    n:
        number of neurons in the network
     p:
        number of memory patterns that the network need to learn
     f :
        the probability in which a neuron will be active in some memory pattern

    Returns:
        the connection matrix, which in the ij coordinate represent the strength of the synapse between the pre-synaptic
        neuron j and the post-synaptic neuron j.
     """
    memory_patterns = _get_p_memory_patterns(f, n, p)
    connection_matrix = calculate_synaptic_strength(memory_patterns, n, f, p)
    return connection_matrix


def find_network_converged_pattern(connection_matrix, start_pattern_vector, t):
    """
    this function calculate a dynamic of an a-synchrony Hopfield network

    Args:

    connection_matrix:
        connection_matrix is a matrix with the synaptic strength of the synapse connecting the
        pre-synaptic j neuron and the post-synaptic i neuron in the ij coordinate. this matrix is with dimensions NXN

    start_pattern_vector:
        start_pattern_vector is a vector representing the start state of the neuronal activity of the
        network. this vector is with dimensions NX1

    t:
        the threshold of the neurons

    Returns:
        the memory pattern that the network converged into
    """
    current_pattern = None
    next_pattern = np.array(start_pattern_vector)
    n = next_pattern.size
    while not np.array_equal(current_pattern, next_pattern):
        current_pattern = np.array(next_pattern)
        for i in range(n):
            hi = np.dot(connection_matrix[i], next_pattern)
            next_pattern[i] = (hi - t) > 0  # like heaviside function
    return next_pattern


def find_percentage_of_activity_changes(connection_matrix, start_pattern_vector, t):
    """
    When the network reaches convergence, we find how many percent of the cells remained in the state in which we
    initialized it.
    this function use find_network_converged_pattern function and calculate the percentage (range [0,1])
    of neurons that their activity changed from the start pattern in the dynamics until convergence

     Args:
    connection_matrix:
        connection_matrix is a matrix with the synaptic strength of the synapse connecting the pre-synaptic j neuron
        and the post-synaptic i neuron in the ij coordinate. this matrix is with dimensions NXN
    start_pattern_vector:
        start_pattern_vector is a vector representing the start state of the neuronal activity of the network. this vector
        is with dimensions NX1
    t:
        t is the threshold of the neurons

    Returns:
        the percentage which have been calculated
    """
    start_pattern = np.array(start_pattern_vector)
    n = start_pattern.size
    converged_pattern = find_network_converged_pattern(connection_matrix, start_pattern, t)
    sum_of_changes = 0
    for i in range(n):
        if start_pattern[i] != converged_pattern[i]:
            sum_of_changes += 1
    percentage = sum_of_changes / n
    return percentage


def create_graph(f):
    plt.title(f"f = {f}")
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.xlabel("a = p/n")
    plt.ylabel("average mistake")


def get_list():
    list_a = []
    for alpha in range(2, 80, 4):
        list_a.append(alpha / 100)
    list_a.append(0.8)
    return list_a


def find_probability_for_mistake_in_memory_convergence():
    """
    this function calculate the probability to converge to a different memory pattern than the one we initialized the
    network with, as a function of a=p/n, a factor of load of the system (proportion between number of memory pattern
    embedded in the system and number of neurons in it), and for different f = probability that a neuron in the network
    will be active. this will show that for bigger a and f, we get a network which does more mistakes
    """
    n = 1000
    list_a = get_list()
    for f in [0.1, 0.2, 0.3]:
        create_graph(f)
        average_a_mistake_list = []
        for a in list_a:
            mistake_percentage_sum_a = 0
            for i in range(5):
                p = math.ceil(a * n)
                memory_patterns = _get_p_memory_patterns(f, n, p)
                connection_matrix = calculate_synaptic_strength(memory_patterns, n, f, p)
                mistake_percentage_sum_a += find_percentage_of_activity_changes(connection_matrix, memory_patterns[0],
                                                                                0.5 - f)
            average_a_mistake_list.append(mistake_percentage_sum_a / 5)
        plt.scatter(list_a, average_a_mistake_list)
        plt.show()
