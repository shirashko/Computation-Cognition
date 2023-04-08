from bellman import *


def main():
    print(value_iteration())
    home_values_TD, out_values_TD = TD_learning_algorithm()
    show_values_progress_in_graph(home_values_TD, out_values_TD, 33 / 19, 161 / 133, 2)
    home_values_Q, out_values_Q = Q_learning_algorithm()
    show_values_progress_in_graph(home_values_Q, out_values_Q, 26 / 9, 4, 3)


if __name__ == "__main__":
    main()
