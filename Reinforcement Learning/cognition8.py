import numpy as np
import matplotlib.pyplot as plt
import random

NUM_OF_EXAMPLES = 12665  # or 6 at the end?
NUM_OF_SYNAPSE = 784
LEARNING_RATE = 0.01


# question 1
def load_data():
    examples = np.loadtxt("Ex8_data.csv", delimiter=",")  # a matrix shape (783, 12665). each example is a vector size
    # 784, so each column in an example.
    labels = np.loadtxt("Ex8_labels.csv", delimiter=",").reshape(-1)
    test_examples = np.loadtxt("Ex8_test_data.csv", delimiter=",")
    test_labels = np.loadtxt("Ex8_test_labels.csv", delimiter=",")
    return examples.transpose(), labels, test_examples.transpose(), test_labels


# question 2 helper
def do_test(test_data, test_labels, weight_vec):
    correct_labels_counter = 0
    for i in range(test_data.shape[0]):  # test_data.shape[0] = 2115 = number of examples in the test
        current_example = test_data[i]
        exp_wx = np.exp(-np.dot(weight_vec, current_example))
        p = 1 / (1 + exp_wx)
        y = random.choices([1, 0], [p, 1 - p])[0]
        if y == test_labels[i]:
            correct_labels_counter += 1
    return correct_labels_counter / test_data.shape[0]


# question 2
def stochastic_binary_perceptron_learning_algorithm(examples_mat, labels_vec, test_data,
                                                    test_labels):
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
            system_accuracy.append(do_test(test_data, test_labels, weight_vec))
    return weight_vec, system_accuracy


# question 3
def show_accuracy(accuracy):
    plt.figure()
    x = [50 * i for i in range(1, len(accuracy) + 1)]
    plt.title("accuracy of the system")
    plt.scatter(x, accuracy, s=0.3, label="accuracy", color="red")
    plt.grid()
    plt.show()


def show_weight_vec_a_image(weight_vector):
    weight_vec = np.reshape(weight_vector, (28, 28))
    plt.imshow(weight_vec, interpolation='nearest')
    plt.show()


def main():
    examples, labels, test_examples, test_labels = load_data()
    weight_vector, accuracy = stochastic_binary_perceptron_learning_algorithm(examples, labels, test_examples,
                                                                              test_labels)
    show_accuracy(accuracy)
    show_weight_vec_a_image(weight_vector)


main()
