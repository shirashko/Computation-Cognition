import numpy as np
import matplotlib.pyplot as plt
import random

NUM_OF_EXAMPLES = 12665
NUM_OF_SYNAPSE = 784
LEARNING_RATE = 0.01


def load_data():
    """ This function load data from MNIST dataset, which used for image recognition and classification tasks in machine
    learning. It consists a set of handwritten digits, each is represented as a grayscale image of size 28x28  pixels.
    The dataset contains a training set and a test set. The digits in the images range from 0 to 9, and each image is
    labeled with the corresponding digit.

    Returns:
        output1: train examples
        output2: train labels
        output 3: test examples
        output 4: test labels
    """
    examples = np.loadtxt("Ex8_data.csv", delimiter=",")  # a matrix shape (783, 12665). each example is a vector with
    # 784 features, so each column in an example.
    labels = np.loadtxt("Ex8_labels.csv", delimiter=",").reshape(-1)
    test_examples = np.loadtxt("Ex8_test_data.csv", delimiter=",")
    test_labels = np.loadtxt("Ex8_test_labels.csv", delimiter=",")
    return examples.transpose(), labels, test_examples.transpose(), test_labels


def _do_test(test_data, test_labels, weight_vec):
    """This function

    Args:
        test_data:
        test_labels:
        weight_vec:

    Returns:
    """
    correct_labels_counter = 0
    for i in range(test_data.shape[0]):  # test_data.shape[0] = 2115 = number of examples in the test
        current_example = test_data[i]
        exp_wx = np.exp(-np.dot(weight_vec, current_example))
        p = 1 / (1 + exp_wx)
        y = random.choices([1, 0], [p, 1 - p])[0]
        if y == test_labels[i]:
            correct_labels_counter += 1
    return correct_labels_counter / test_data.shape[0]


def stochastic_binary_perceptron_learning_algorithm(examples_mat, labels_vec, test_data, test_labels):
    """This function

    Args:
        examples_mat:
        labels_vec:
        test_data:
        test_labels:

    Returns:
        weight_vec:
        system_accuracy:
    """
    # each row is an example
    weight_vec = np.random.normal(0, 0.001, 784)
    counter = 50
    system_accuracy = []
    for i in range(NUM_OF_EXAMPLES):
        counter += 1
        current_example = examples_mat[i]
        exp_wx = np.exp(-np.dot(weight_vec, current_example))
        p = 1 / (1 + exp_wx)
        y = random.choices([1, 0], [p, 1 - p])[0]
        if y == labels_vec[i]:
            reward = 1
        else:
            reward = 0
        for j in range(NUM_OF_SYNAPSE):
            learning_update_step = LEARNING_RATE * reward * (y - p) * current_example[j]
            weight_vec[j] += learning_update_step
        if counter % 50 == 0:  # checks the accuracy of the system on the test data
            system_accuracy.append(_do_test(test_data, test_labels, weight_vec))
    return weight_vec, system_accuracy


def show_accuracy(accuracy):
    """This function show the accuracy of a system

    Args:
        accuracy (list): the accuracy of the system
    """
    plt.figure()
    x = [50 * i for i in range(1, len(accuracy) + 1)]
    plt.title("accuracy of the system")
    plt.scatter(x, accuracy, s=0.3, label="accuracy", color="red")
    plt.grid()
    plt.show()


def show_weight_vec_as_image(weight_vector):
    """ This function show an image that represent a given weight vector

    Args:
        weight_vector: the weight vector to show image of
    """
    weight_vec = np.reshape(weight_vector, (28, 28))
    plt.imshow(weight_vec, interpolation='nearest')
    plt.show()
