import numpy as np
import math
from numpy.linalg import inv
from matplotlib import pyplot as plt


def create_examples(P):
    examples_matrix = []
    for i in range(P):
        examples_matrix.append([np.random.uniform(-5, 5), 1])
    return examples_matrix


def Y0(x):
    return 1 + x + math.pow(x, 2) + math.pow(x, 3)


def create_labels(P, examples):
    labels = []
    for i in range(P):
        labels.append(Y0(examples[i][0]))
    return labels


def calculate_generalization_error(W):
    x_list = []
    for i in range(-500, 501, 1):
        x_list.append(i / 100)
    result = 0
    for x in x_list:
        y = np.dot([x, 1], W)
        y0 = Y0(x)
        error = y - y0
        result += math.pow(error, 2)
    const = 1 / (2 * len(x_list))
    result *= const
    return result


def calculate_training_error(P, examples_matrix, labels, W):
    const = 1 / (2 * P)
    Y = np.matmul(examples_matrix, W)
    result = 0
    for i in range(P):
        error = Y[i] - labels[i]
        result += math.pow(error, 2)
    result *= const
    return result


def calculate_momentary_error(example, y0, weight_vec):
    y = np.dot(example, weight_vec)
    error = y - y0
    result = 0.5 * math.pow(error, 2)
    return result


def calculate_gradient_vec(P, examples, weight_vec, labels):
    gradient_1 = 0
    gradient_2 = 0
    for i in range(P):
        gradient_1 += 1 / P * (examples[i][0] * (examples[i][0] * weight_vec[0] + weight_vec[1] - labels[i]))
        gradient_2 += 1 / P * (examples[i][0] * weight_vec[0] + weight_vec[1] - labels[i])
    gradient = np.array([gradient_1, gradient_2])
    return gradient


def calculate_gradient_vec_online(example, weight_vec, y0):
    gradient_1 = example * (example * weight_vec[0] + weight_vec[1] - y0)
    gradient_2 = (example * weight_vec[0] + weight_vec[1] - y0)
    return np.array([gradient_1, gradient_2])


def gradient_batch_learning_algorithm(learning_rate, examples, num_of_update_steps, P, labels):
    """
    batch gradient learning

    Args:
        learning_rate:

        examples:

        num_of_update_steps:

        P:

        labels:

    Returns:
    """
    g_err_list = []
    t_err_list = []
    weight_vec = np.array([1, 1])  # initialize with arbitrary numbers
    for j in range(num_of_update_steps):
        g_err_list.append(calculate_generalization_error(weight_vec))
        t_err_list.append(calculate_training_error(P, examples, labels, weight_vec))
        gradient_vec = calculate_gradient_vec(P, examples, weight_vec, labels)
        weight_vec = weight_vec - (learning_rate * gradient_vec)
    return t_err_list, g_err_list


def gradient_online_learning_algorithm(learning_rate, examples, num_of_update_steps, labels):
    """
    on-line gradient learning

    Args:
        learning_rate:

        examples:

        num_of_update_steps:

        labels:

    Returns:
    """
    g_err_list = []
    m_err_list = []  # m for moment
    weight_vec = np.array([1, 1])  # initialize with arbitrary numbers
    for j in range(num_of_update_steps):
        g_err_list.append(calculate_generalization_error(weight_vec))
        m_err_list.append(calculate_momentary_error(examples[j], labels[j], weight_vec))
        gradient_vec = calculate_gradient_vec_online(examples[j][0], weight_vec, labels[j])
        weight_vec = weight_vec - (learning_rate * gradient_vec)
    return m_err_list, g_err_list


def calculate_input_correlation_matrix(P, examples_matrix):
    """
    calculate correlation matrix

    Args:
        P:

        examples_matrix:

    Returns:
    """
    # Cij = 1/p * ( sigma ( xi (m) * xj (m) , m=1,...P)
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


def calculate_correlation_input_output_vector(P, examples_matrix, labels_vector):
    # ui =  1/p * ( sigma ( xi(m) * y0(m) , m=1,...P)
    const = 1 / P
    u1 = 0
    for m in range(P):
        u1 += examples_matrix[m][0] * labels_vector[m]
    u2 = np.sum(labels_vector)  # multiply each coordinate by 1 = X2 won't change anything
    U = np.array([u1, u2]) * const
    return U


def correlation_matrix_reversal_algorithm(P, examples, labels):
    C = calculate_input_correlation_matrix(P, examples)
    U = calculate_correlation_input_output_vector(P, examples, labels)
    W = np.matmul(inv(C), U)
    t_err = calculate_training_error(P, examples, labels, W)
    g_err = calculate_generalization_error(W)
    return t_err, g_err


def show_results(batch_err_t, batch_err_g, online_err_m, online_err_g, invert_err_t, invert_err_g):
    plt.figure()
    plt.title("generalization, training and temporary errors of different algorithms")
    plt.plot(range(100), batch_err_t, label="Batch algorithm training error", color="red", linewidth=1)
    plt.plot(range(100), online_err_m, label="On-line algorithm momentary error", color="blue", linewidth=1)
    plt.axhline(invert_err_t, color="green", linestyle="-", label="inverse algorithm training error", linewidth=1)
    plt.plot(range(100), batch_err_g, label="Batch algorithm generalization error", color="purple", linewidth=1)
    plt.plot(range(100), online_err_g, label="On-line algorithm generalization error", color="black", linewidth=1)
    plt.axhline(invert_err_g, color="orange", linestyle="-", label="inverse algorithm generalization error",
                linewidth=1)
    plt.xlabel("number of updating step")
    plt.ylabel("error")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()


def show_graph_of_batch_t_err(t_err_batch):
    plt.figure()
    plt.plot(range(500), t_err_batch[0], label="learning rate = 0.002", color="red", linewidth=1)
    plt.plot(range(500), t_err_batch[1], label="learning rate = 0.005", color="blue", linewidth=1)
    plt.plot(range(500), t_err_batch[2], label="learning rate = 0.01", color="green", linewidth=1)
    plt.plot(range(500), t_err_batch[3], label="learning rate = 0.02", color="orange", linewidth=1)
    plt.plot(range(500), t_err_batch[4], label="learning rate = 0.05", color="purple", linewidth=1)
    plt.xlabel("number of updating step")
    plt.ylabel("error")
    plt.legend(loc="upper right")
    plt.grid()
    plt.title("training error of batch algorithm with different learning rates")
    plt.show()


def show_graph_of_batch_g_err(g_err_batch):
    plt.figure()
    plt.plot(range(500), g_err_batch[0], label="learning rate = 0.002", color="red", linewidth=1)
    plt.plot(range(500), g_err_batch[1], label="learning rate = 0.005", color="blue", linewidth=1)
    plt.plot(range(500), g_err_batch[2], label="learning rate = 0.01", color="green", linewidth=1)
    plt.plot(range(500), g_err_batch[3], label="learning rate = 0.02", color="orange", linewidth=1)
    plt.plot(range(500), g_err_batch[4], label="learning rate = 0.05", color="purple", linewidth=1)
    plt.xlabel("number of updating step")
    plt.ylabel("error")
    plt.legend(loc="upper right")
    plt.grid()
    plt.title("generalization error of batch algorithm with different learning rates")
    plt.show()


def show_graph_of_online_err(g_err_online):
    plt.figure()
    plt.plot(range(500), g_err_online[0], label="learning rate = 0.002", color="red", linewidth=1)
    plt.plot(range(500), g_err_online[1], label="learning rate = 0.005", color="blue", linewidth=1)
    plt.plot(range(500), g_err_online[2], label="learning rate = 0.01", color="green", linewidth=1)
    plt.plot(range(500), g_err_online[3], label="learning rate = 0.02", color="orange", linewidth=1)
    plt.plot(range(500), g_err_online[4], label="learning rate = 0.05", color="purple", linewidth=1)
    plt.xlabel("number of updating step")
    plt.ylabel("error")
    plt.legend(loc="upper right")
    plt.grid()
    plt.title("generalization error of online algorithm with different learning rates")
    plt.show()


