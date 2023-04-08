import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA

NUM_OF_SIMULATIONS = 100
LEARNING_RATE = 1


def find_weight_vector(examples_matrix, examples_labels):
    """
    This function find a weight vector that classify correctly the given train set using the Perceptron learning
    algorithm

    Args:
        examples_matrix (NXP matrix): matrix with P examples (from Rn) to classify
        examples_labels (PX1 matrix): the right labels for the given examples

    Returns:
        weight vector (NX1 matrix)
    """
    # initialization stage
    N_dimension, p_examples = examples_matrix.shape[0], examples_matrix.shape[1]
    weight_vector = np.ones(N_dimension, dtype=int)
    # iteration stage
    have_mistake = True
    while have_mistake:
        have_mistake = False
        for i in range(p_examples):
            current_col_as_row = examples_matrix[:, i]  # arr[:, i] create np.array with the values of the i'th column
            current_label = examples_labels[i]
            result = np.dot(current_col_as_row, weight_vector)
            binary_result = 0
            if result > 0:
                binary_result = 1
            if binary_result != current_label:
                have_mistake = True
                weight_vector = np.add(weight_vector, LEARNING_RATE * (2 * current_label - 1) * current_col_as_row)
    return weight_vector


def _creating_labels(examples_matrix):
    """
    creating the examples_labels by the directions: if x1 > x2 then y = 1, otherwise y = 0

    Args:
        examples_matrix:
            the examples to label

    Returns:
        a label vector
    """
    p = examples_matrix.shape[1]
    examples_labels = np.zeros(p, dtype=int)
    examples_labels[examples_matrix[0, :] > examples_matrix[1, :]] = 1
    return examples_labels


def _present_weight_vector_and_separate_line(weight_vector):
    """
    present a given weight vector and the separate line which is vertical to it in a graph

    Args:
        weight_vector: the weight vector to present
    """
    slope = weight_vector[1] / weight_vector[0]  # y\x
    separate_line_slope = -1 / slope  # m1 * m2 = -1 for vertical lines with slopes m1,m2
    x = np.arange(-10, 11)
    y1 = slope * x
    y2 = separate_line_slope * x
    plt.plot(x, y1)
    plt.plot(x, y2)


def perceptron_algorithm_in_action():
    """
    creating 1000 random examples from U[-10,10] distribution, then finding the correct labels to these examples
    when the separating line is y=x. find a weight vector using the perceptron learning algorithm, then present the
    examples with the label they got using the returned w, and also present the weight vector and the separate line to
    see if the examples got the right classification
    """
    examples_matrix = np.random.uniform(-10, 10, size=(2, 1000))
    # creating the examples_labels by the directions: if x1 > x2 then y = 1, otherwise y = 0
    examples_labels = _creating_labels(examples_matrix)
    weight_vector = find_weight_vector(examples_matrix, examples_labels)
    # find the result vector using the weight vector returned from the find_weight_vector algorithm by multiplying the
    # examples with the weight vector to get the result vector, and then put it as an input on the H function that
    # return 1 if > 0, otherwise return 0
    transposed_matrix = np.transpose(examples_matrix)
    result = np.matmul(transposed_matrix, weight_vector)
    H_res = np.zeros_like(result)
    H_res[result > 0] = 1
    plt.scatter(x=examples_matrix[0, :], y=examples_matrix[1, :], c=examples_labels, cmap='Set1', s=4)
    plt.title("examples colored by the classification calculated with the perceptron algorithm, for underlying rule y="
              "x", fontsize=9)
    _present_weight_vector_and_separate_line(weight_vector)
    plt.show()


def calculate_angle(v1, v2):
    """ calculate angle between two vectors. v1*v2 = norm(v1) * norm(v2) * cos(angle), where angle represent the angle
    between v1 and v2. we will choose the min from a,180-a because we care about the angle between the lines that
    are represented by the vectors, and not about the angle between the vectors."""
    cos_of_the_angle = np.dot(v1, v2) / (LA.norm(v1) * LA.norm(v2))
    angle = np.rad2deg(np.arccos(cos_of_the_angle))
    the_complement_angle = 180 - angle
    min_angle = min(the_complement_angle, angle)
    return min_angle


# create graph for question 4
def create_graph():
    plt.title(f"Average Error As Function Of Train Set Size\n error = absolute value of the angle between optimal w "
              f"and perceptron algorithm vector", fontsize=10)
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    plt.xlim(0, 520)
    plt.ylim(0, 90)
    plt.grid()
    plt.xlabel("number of examples")
    plt.ylabel("average mistake")


def average_mistake_as_function_of_sample_size():
    """
    checking for P = 5, 20, 30, 50, 100, 150, 200, 500 random examples the perceptron learning algorithm for finding
    a correct weight_vector. for each P check the average "error" between the optimal weight_vector to the
    weight_vector which the perceptron learning algorithm found by simulating 100 trials. at last, present in a graph
    the average error as a function of P (P = number of examples sent to the perceptron learning algorithm)
    the optimal separate line in our case is y = x so the optimal weight vector in our case is on the line y = -x,
    so weight_vector = (x,-x) (for example (-1,1)) is the optimal solution.
    """
    optimal_vector = np.array([1, -1])
    num_of_example_list = [1, 2, 5, 20, 30, 50, 100, 150, 200, 500]
    average_mistake = []
    for p in num_of_example_list:
        sum_of_mistakes = 0
        for i in range(NUM_OF_SIMULATIONS):
            examples_matrix = np.random.uniform(-10, 10, size=(2, p))
            examples_labels = _creating_labels(examples_matrix)
            weight_vector = find_weight_vector(examples_matrix, examples_labels)
            sum_of_mistakes += abs(calculate_angle(weight_vector, optimal_vector))
        # calculate the average mistake of the 100 simulations
        average_mistake_p = sum_of_mistakes / NUM_OF_SIMULATIONS
        average_mistake.append(average_mistake_p)
    create_graph()
    plt.scatter(num_of_example_list, average_mistake, s=15)
    plt.show()
