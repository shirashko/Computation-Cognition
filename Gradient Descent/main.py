from gradient_descent import *


def main():
    # part 1
    examples = create_examples(300)
    labels = create_labels(100, examples)
    x1, x2 = gradient_batch_learning_algorithm(0.01, examples, 100, 100, labels)
    y1, y2 = gradient_online_learning_algorithm(0.01, examples, 100, labels)
    z1, z2 = correlation_matrix_reversal_algorithm(100, examples, labels)
    show_results(x1, x2, y1, y2, z1, z2)

    # part 2
    examples = create_examples(500)
    labels = create_labels(500, examples)
    learning_rates = [0.002, 0.005, 0.01, 0.02, 0.05]
    t_err_batch_list = []  # list of lists
    g_err_batch_list = []
    g_err_online_list = []
    for learning_rate in learning_rates:
        t_err_batch, g_err_batch = gradient_batch_learning_algorithm(learning_rate, examples, 500, 500, labels)
        t_err_batch_list.append(t_err_batch)
        g_err_batch_list.append(g_err_batch)
        m_err_online, g_err_online = gradient_online_learning_algorithm(learning_rate, examples, 500, labels)
        g_err_online_list.append(g_err_online)

    show_graph_of_batch_t_err(t_err_batch_list)
    show_graph_of_batch_g_err(g_err_batch_list)
    show_graph_of_online_err(g_err_online_list)


if __name__ == "__main__":
    main()
