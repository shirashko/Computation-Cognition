from decisions_under_uncertainty import *


def main():
    # part 1
    list_of_xs = [question1_round(1)]
    for i in range(6):
        cur_xs = question1_round(list_of_xs[i])
        list_of_xs.append(cur_xs)
    print(list_of_xs)
    question1_show_graph(list_of_xs, [0.5 ** i for i in range(1, 8)])
    show_utility_function_for_different_sigmas()
    show_pi_function_for_different_alphas()

    # part 2
    data_matrix = np.loadtxt("ex10_q2_data.csv", delimiter=",", skiprows=1)
    data_matrix[:, 2] = data_matrix[:, 2] / 100  # convert percentage into probability in range [0,1]
    first_trial_data = data_matrix[data_matrix[:, NUMBER_OF_TRIAL_COL] == 1]
    sec_trial_data = data_matrix[data_matrix[:, NUMBER_OF_TRIAL_COL] == 2]
    alpha_1, sigma_1 = find_sigma_alpha(first_trial_data)
    alpha_2, sigma_2 = find_sigma_alpha(sec_trial_data)
    show_graph_comparing_trials(alpha_1, alpha_2, "alpha")
    show_graph_comparing_trials(sigma_1, sigma_2, "sigma")
    print(np.mean(alpha_1), np.mean(sigma_1))


if __name__ == "__main__":
    main()
