import numpy
import numpy as np
import random
import matplotlib.pyplot as plt


def get_random_example(p, f, sigma):
    x1 = np.random.uniform(-1, 1)
    epsilon = np.random.normal(0, sigma ** 2)
    x2 = random.choices([np.sin(f * x1) + epsilon, np.random.uniform(-1, 1)], [p, 1 - p])[0]
    return [x1, x2]


def create_prototypes():
    return np.random.uniform(-1, 1, (100, 2))


def pi_func(k):
    # this function do operation in coordinate coordinate way. instead calculate to prototype separately I do it
    # together
    ind = np.array(range(1, 101))
    pifunc = np.exp(
        (-(ind - k) ** 2 / (2 * 4 ** 2)))  # a vector 1X100 of the pik(l-k)/Ck for each l for prototype index
    Ck = 1 / sum(pifunc)
    pifunc = Ck * pifunc  # a vector 1X100 of the pik(l-k) for each l for prototype index
    return pifunc


def SOM():
    prot_matrix = create_prototypes()  # 100 X 2
    first_prot = numpy.array(prot_matrix)
    list_of_examples = []  # 100 X 2
    for i in range(20000):
        curr_example = get_random_example(0.95, 4, 0.1)
        list_of_examples.append(curr_example)
        sub_matrix = curr_example - prot_matrix  # 100 X 2
        norm_vec = np.linalg.norm(sub_matrix, axis=1)  # normalize each line in individual way. we get 1X100 vec of
        # norms of each layer
        min_l = np.argmin(norm_vec)  # min_l == layer that have the most similar prototype to curr_example
        # update the prototypes
        pi_vec = pi_func(min_l)  # 1 X 100
        pi_mat = np.array([pi_vec, pi_vec]).transpose()  # 100 x 2
        prot_matrix += sub_matrix * pi_mat
    return first_prot, prot_matrix, list_of_examples


def create_graphs():  # check if the x's and the y's I gave is the correct ones
    first_prot, last_prot, list_of_examples = SOM()  # 100 X 2
    plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    plt.title("SOM examples with first arbitrary prototype set")
    plt.scatter(np.transpose(list_of_examples)[0], np.transpose(list_of_examples)[1], s=0.3,
                label="examples", color="green", linewidth=0.5)
    plt.plot(first_prot.transpose()[0], first_prot.transpose()[1], '.--', label="prototypes", color="blue", linewidth=1,
             alpha=0.3)
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()

    # second graph
    plt.figure()
    plt.title("SOM examples with learned prototype set")
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    plt.scatter(np.transpose(list_of_examples)[0], np.transpose(list_of_examples)[1], s=0.3,
                label="examples", color="green", linewidth=0.5)
    plt.plot(last_prot.transpose()[0], last_prot.transpose()[1], '.--', label="prototypes", color="blue", linewidth=1,
             alpha=0.3)
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()


create_graphs()
