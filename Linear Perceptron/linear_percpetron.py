import numpy as np
import math
from numpy.linalg import inv
from matplotlib import pyplot as plt

X2 = 1
FIRST_COORDINATE = 0
SECOND_COORDINATE = 1
NUM_OF_SIMULATIONS = 100


# for question 1
def create_examples(P):
    examples_matrix = []  # dimensions 500X2 = PX2
    for m in range(P):
        examples_matrix.append(np.array([np.random.uniform(-1, 1), X2]))
    return examples_matrix
    # return np.random.uniform(-1, 1, size=(P, X2))


# for question 1
def find_labels(P, examples_matrix):
    # calculate the labels for the function x^3 - x^2
    labels = []  # dimensions 500X1
    for m in range(P):
        x = examples_matrix[m][FIRST_COORDINATE]
        labels.append(math.pow(x, 3) - math.pow(x, 2))
    return labels


# question 1
def question1(P):
    examples_matrix = create_examples(P)
    labels = find_labels(P, examples_matrix)
    return examples_matrix, labels


def calculate_input_correlation_matrix(P, examples_matrix):
    # Cij = 1/p * ( sigma ( xi (m) * xj (m) , m=1,...P)
    const = 1 / P
    C11 = 0
    for m in range(P):
        C11 += math.pow(examples_matrix[m][FIRST_COORDINATE], 2)
    C12 = 0
    for m in range(P):
        C12 += examples_matrix[m][FIRST_COORDINATE]  # multiply by 1 = X2 won't change anything
    C21 = C12  # symmetric
    C22 = P  # summing 1 P times == P
    C = np.array(([C11, C21], [C21, C22])) * const
    return C


def calculate_correlation_input_output_vector(P, examples_matrix, labels_vector):
    # ui =  1/p * ( sigma ( xi(m) * y0(m) , m=1,...P)
    const = 1 / P
    u1 = 0
    for m in range(P):
        u1 += examples_matrix[m][FIRST_COORDINATE] * labels_vector[m]
    u2 = np.sum(labels_vector)  # multiply each coordinate by 1 = X2 won't change anything
    U = np.array([u1, u2]) * const
    return U


def question2(P, examples_matrix, labels_vector):
    C = calculate_input_correlation_matrix(P, examples_matrix)
    U = calculate_correlation_input_output_vector(P, examples_matrix, labels_vector)
    W = np.matmul(inv(C), U)
    return W


def create_graph():
    plt.title(f"wanted output versa Perceptron output")
    plt.figtext(0.14, 0.8, "blue: desire output\nred: Perceptron output")
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")


# question 3
def question3(P):
    examples_matrix, labels = question1(P)
    create_graph()
    for i in range(P):
        plt.plot(examples_matrix[i][0], labels[i], '.' + "b")
    W = question2(P, examples_matrix, labels)
    for i in range(P):
        y = examples_matrix[i][0] * W[0] + W[1]
        plt.plot(examples_matrix[i][0], y, '.' + "r")
    plt.show()


def calculate_training_error(P, examples_matrix, labels):
    const = 1 / (2 * P)
    W = question2(P, examples_matrix, labels)
    Y = np.matmul(examples_matrix, W)
    result = 0
    for i in range(P):
        error = Y[i] - labels[i]
        result += math.pow(error, 2)
    result *= const
    return result


def calculate_generalization_error(P, examples_matrix, labels):
    x_list = []
    for i in range(-100, 100, 1):
        x_list.append(i / 1000)
    W = question2(P, examples_matrix, labels)
    result = 0
    for x in x_list:
        y = np.dot([x, 1], W)
        y0 = math.pow(x, 3) - math.pow(x, 2)
        error = y - y0
        result += math.pow(error, 2)
    const = 1 / (2 * len(x_list))
    result *= const
    return result


# question 4
def question4(P):
    examples_matrix, labels = question1(P)
    t_err = calculate_training_error(P, examples_matrix, labels)
    g_err = calculate_generalization_error(P, examples_matrix, labels)
    return t_err, g_err


def show_graph_question5(p_list, t_err_list, g_err_list):
    plt.title("question 5: average of training and generalization errors as a function of P")
    plt.xlim(0, 110)
    plt.ylim(0, 0.2)
    plt.grid()
    plt.xlabel("P")
    plt.ylabel("error")
    plt.figtext(0.14, 0.8, "blue: training error\nred: generalization error")
    for i in range(len(p_list)):
        plt.plot(p_list[i], t_err_list[i], '.' + "b")
        plt.plot(p_list[i], g_err_list[i], '.' + "r")
    plt.show()


def question5():
    p_list = [i for i in range(5, 101, 5)]
    t_err_list = []
    g_err_list = []
    for P in p_list:
        sum_t = 0
        sum_g = 0
        for i in range(NUM_OF_SIMULATIONS):
            t_err, g_err = question4(P)
            sum_t += t_err
            sum_g += g_err
        average_t_for_P = sum_t / NUM_OF_SIMULATIONS
        average_g_for_P = sum_g / NUM_OF_SIMULATIONS
        t_err_list.append(average_t_for_P)
        g_err_list.append(average_g_for_P)

    show_graph_question5(p_list, t_err_list, g_err_list)
