from reinforcement_learning import *


def main():
    examples, labels, test_examples, test_labels = load_data()
    weight_vector, accuracy = stochastic_binary_perceptron_learning_algorithm(examples, labels, test_examples,
                                                                              test_labels)
    show_accuracy(accuracy)
    show_weight_vec_as_image(weight_vector)


if __name__ == "__main__":
    main()
