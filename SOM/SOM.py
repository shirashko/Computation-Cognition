import numpy
import numpy as np
import random
import matplotlib.pyplot as plt


def get_random_example(p, f, sigma):
    """This function create 2-dimensions random example, first coordinate x1 from U[-1,1], second coordinate x2 in
    probability p is sin(f*x1)+e when e is a random sample from N(0,sigma^2) distribution, and in the complement
    probability x2 derive from U[-1,1] distribution independently of x1

    Args:
        p: the probability to choose x2 to be in the form sin(f*x1)+e
        f: the constant we multiply by x1 when x2 is from the form sin(f*x1)+e
        sigma: the std of the distribution we derive the noise e

    Returns:
        list represent the random example
    """
    x1 = np.random.uniform(-1, 1)
    epsilon = np.random.normal(0, sigma ** 2)
    x2 = random.choices([np.sin(f * x1) + epsilon, np.random.uniform(-1, 1)], [p, 1 - p])[0]
    return [x1, x2]



def pi_func(k):
    """ This function do operation in coordinate way. instead calculate to prototype separately I do it together
    Args:

    Returns:

    """
    ind = np.array(range(1, 101))
    pifunc = np.exp(
        (-(ind - k) ** 2 / (2 * 4 ** 2)))  # a vector 1X100 of the pik(l-k)/Ck for each l for prototype index
    Ck = 1 / sum(pifunc)
    pifunc = Ck * pifunc  # a vector 1X100 of the pik(l-k) for each l for prototype index
    return pifunc


def SOM():
    """ This function

    Returns:
        first_prot:
        prot_matrix:
        list_of_examples:
    """
    prot_matrix = np.random.uniform(-1, 1, (100, 2))  # create prototypes
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


def create_graphs():
    """This function

    """
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

