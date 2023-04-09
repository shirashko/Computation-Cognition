import numpy as np
import math
from numpy.linalg import inv
from matplotlib import pyplot as plt

NUM_OF_SIMULATIONS = 100


def _create_examples(P):
    """
    create PX2 matrix of random numbers from U(-1,1) distribution, when the second feature is always 1

    Args:
        P (int): Number of examples to create

    Returns:
        The examples' matrix, size PX2
    """
    return np.array([[np.random.uniform(-1, 1), 1] for m in range(P)])


def _labels_function(x):
    """ This function is the implementation of the function x ** 3 - x ** 2
    Args:
        x (PX1 array): list of all the x to find the corresponding label for
    """
    return x ** 3 - x ** 2


def _find_labels(examples_matrix):
    """
    this function calculates for a matrix with P examples, by the first coordinate of each example, the value of
    x^3 - x^2, and returns the results as a list in the corresponding order

    Args:
        examples_matrix: a PX2 matrix which it's columns are the examples to calculate the labels for
    """
    examples = examples_matrix[:, 0]
    return _labels_function(examples)


def create_examples_and_true_labels(P):
    """
    this function create P examples, each example with two coordinates, first random from uniform distribution range
    (-1,1), second is always 1. additionally, it finds the true labels according to the rule x^3 - x^2

    Aegs:
        P (int): number of examples to create

    Returns:
        example_matrix (PX2 ndarray): the design matrix
        labels (PX1 ndarray): the response vector

    """
    examples_matrix = _create_examples(P)
    labels = _find_labels(examples_matrix)
    return examples_matrix, labels


def _calculate_input_correlation_matrix(examples_matrix):
    """ This function calculate the input correlation matrix (for P examples) needed computing the weight vector

        Args:
            examples_matrix (PX2 ndarray): the columns of the matrix is the examples

        Returns:
            the correlation matrix (2X2 ndarray)
    """
    # Cij = 1/p * ( sigma ( xi (m) * xj (m) , m=1,...P)
    P = examples_matrix.shape[0]
    const = 1 / P
    C11 = 0
    for m in range(P):
        C11 += math.pow(examples_matrix[m][0], 2)
    C12 = 0
    for m in range(P):
        C12 += examples_matrix[m][0]  # multiply by 1 = X2 won't change anything
    C21 = C12  # symmetric
    C22 = P  # summing 1 P times == P
    C = np.array(([C11, C21], [C21, C22])) * const
    return C


def _calculate_correlation_input_output_vector(examples_matrix, labels_vector):
    """ This function calculate the input output correlation vector needed computing the weight vector

    Args:
        examples_matrix (PX2 ndarray): the columns of the matrix is the examples
        labels_vector (PX1 ndarray): the response vector

    Returns:
        the input output correlation vector (2X1 ndarray)
    """
    # ui =  1/p * ( sigma ( xi(m) * y0(m) , m=1,...P)
    P = examples_matrix.shape[0]
    const = 1 / P
    u1 = 0
    for m in range(P):
        u1 += examples_matrix[m][0] * labels_vector[m]
    u2 = np.sum(labels_vector)  # multiply each coordinate by 1 = X2 won't change anything
    U = np.array([u1, u2]) * const
    return U


def find_weight_vector(examples_matrix, labels_vector):
    """
    this function calculate the correlation matrix of the inputs, the correlation vector between the input and the
    output, and using it finds a suitable weight vector

    Args:
        examples_matrix (PX2 ndarray): the matrix of examples
        labels_vector (Px1 ndarray): the true labels

    Returns:
        A weight vector for labeling future examples
    """
    C = _calculate_input_correlation_matrix(examples_matrix)
    U = _calculate_correlation_input_output_vector(examples_matrix, labels_vector)
    W = np.matmul(inv(C), U)
    return W


def _create_graph(x, y_real, y_predict, W):
    """ This function present the real function and the learned function in a graph
        Args:
            x (1XP array):
                list of x for the range we want to display
            y_real (1XP array):
                the y list matching to the real function
            t_predict (1XP array):
                the y list matching to the learned function
            W (2X1 array):
                the weight vector which we learned
    """
    plt.scatter(x, y_real, label="Real Function: y= x^3 - x^2")
    plt.scatter(x, y_predict, label=f"learned function: y= {round(W[0], 2)}x{round(W[1], 2)}")
    plt.title(f"Real Function Versa Prediction Function")
    plt.grid(), plt.legend()
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)


def show_difference_between_prediction_and_real_functions(P):
    """ This function create a matrix with P examples and calculate the response vector according to the rule: x^3 - x^2
    and then learn a weight vector using batch supervised learning. the learning principle the function use to find a
    good weight vector is multiplying the inverse of the input correlation matrix with the input output correlation
    vector.

    Args:
        P (int):
            number of examples, size of train set
    """
    examples_matrix, labels = create_examples_and_true_labels(P)
    W = find_weight_vector(examples_matrix, labels)
    _create_graph(examples_matrix[:, 0], labels, (examples_matrix[:, 0] * W[0]) + W[1], W)
    plt.show()


def _calculate_training_error(P, examples_matrix, labels):
    const = 1 / (2 * P)
    W = find_weight_vector(examples_matrix, labels)
    Y = np.matmul(examples_matrix, W)
    result = 0
    for i in range(P):
        error = Y[i] - labels[i]
        result += math.pow(error, 2)
    result *= const
    return result


def _calculate_generalization_error(P, examples_matrix, labels):
    x_list = []
    for i in range(-100, 100, 1):
        x_list.append(i / 1000)
    W = find_weight_vector(examples_matrix, labels)
    result = 0
    for x in x_list:
        y = np.dot([x, 1], W)
        y0 = math.pow(x, 3) - math.pow(x, 2)
        error = y - y0
        result += math.pow(error, 2)
    const = 1 / (2 * len(x_list))
    result *= const
    return result


def calculate_train_and_generalization_errors(P):
    examples_matrix, labels = create_examples_and_true_labels(P)
    t_err = _calculate_training_error(P, examples_matrix, labels)
    g_err = _calculate_generalization_error(P, examples_matrix, labels)
    return t_err, g_err


def _show_graph(p_list, t_err_list, g_err_list):
    plt.title("Average Train And Generalization Errors As Function Of Train Size")
    plt.xlabel("number of examples in train set")
    plt.ylabel("error")
    plt.scatter(p_list, t_err_list, s=7, label="train error")
    plt.scatter(p_list, g_err_list, s=7, label="generalization error")
    plt.legend(), plt.grid()
    plt.show()


def show_train_and_generalization_errors():
    """
    Present a graph of the average train and generalization error as a function of the train set size
    """
    p_list = [i for i in range(5, 101, 5)]
    t_err_list, g_err_list = [], []
    for P in p_list:
        sum_t, sum_g = 0, 0
        for i in range(NUM_OF_SIMULATIONS):
            t_err, g_err = calculate_train_and_generalization_errors(P)
            sum_t += t_err
            sum_g += g_err
        average_t_for_P = sum_t / NUM_OF_SIMULATIONS
        average_g_for_P = sum_g / NUM_OF_SIMULATIONS
        t_err_list.append(average_t_for_P)
        g_err_list.append(average_g_for_P)

    _show_graph(p_list, t_err_list, g_err_list)
