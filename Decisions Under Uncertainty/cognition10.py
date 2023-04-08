import HamsterStudentV2
import matplotlib.pyplot as plt
import numpy as np

ID = 209  # 013 invalid so took the first digit of my id
CHOICE_COL = 0
NUMBER_OF_TRIAL_COL = 1
P_COL = 2
SUBJECT_COL = 3
XG_COL = 4
XS_COL = 5
NUMBER_OF_SUBJECTS = 30
GAMBLE = 1
SAFE = 2
LINEAR = 1


# HamsterStudent.myHamster() is a function that get:  (number of pinuts in the safe choice, number of pinuts in the
# gambling choice, 3 digit which represent the id of the hamster)


# find the number of grams I need to give to the my mice for him not have preference to the choice - safe / gambling,
# in case gambling gives him xg (the argument of the function) grams in half a chance and 0 grams on the other half
# chance

def question1_round(xg):
    # search the first point in which the hamster refer the other option
    next_p = cur_p = HamsterStudentV2.myHamster(0, xg, ID)  # 0 for safe option, 1 for gambling option, p for preference
    xs = 0.01
    while cur_p == next_p:
        cur_p = next_p
        xs += 0.01
        next_p = HamsterStudentV2.myHamster(xs, xg, ID)
    return xs


def question1_show_graph(x, y):
    plt.figure()
    plt.title("209")
    plt.plot(x, y, color="blue", label="hamster")
    # plt.plot(x, x, color="red", label="y=x function for comparison")
    plt.xlabel("grams of peanuts")
    plt.ylabel("utility function")
    plt.legend()
    plt.grid()
    plt.show()


def alpha_is_bigger_than_one_pi(p):
    return np.exp(-((-np.log(p)) ** 2))


def alpha_is_one_pi(p):
    return p


def alpha_is_less_than_one_pi(p):
    return np.exp(-((-np.log(p)) ** 0.5))


def show_pi_function_for_different_alphas():
    p = np.linspace(0.01, 1, 100)
    y1 = alpha_is_less_than_one_pi(p)
    y2 = alpha_is_one_pi(p)
    y3 = alpha_is_bigger_than_one_pi(p)
    plt.figure()
    plt.title("pi function with different alphas")
    plt.plot(p, y1, color="green", label="alpha<1")
    plt.plot(p, y2, color="red", label="alpha=1")
    plt.plot(p, y3, color="blue", label="alpha>1")
    plt.legend()
    plt.grid()
    plt.show()


def alpha_is_bigger_than_one_u(p):
    return p ** 2


def alpha_is_one_u(p):
    return p


def alpha_is_less_than_one_u(p):
    return p ** 0.5


def show_utility_function_for_different_sigmas():
    x = np.linspace(0, 100, 1000)
    y1 = alpha_is_less_than_one_u(x)
    y2 = alpha_is_one_u(x)
    y3 = alpha_is_bigger_than_one_u(x)
    plt.figure()
    plt.title("utility function with different sigmas")
    plt.plot(x, y1, color="green", label="sigma<1")
    plt.plot(x, y2, color="red", label="sigma=1")
    plt.plot(x, y3, color="blue", label="sigma>1")
    plt.legend()
    plt.grid()
    plt.show()


def for_x(p):
    return np.log(-np.log(p))


def for_y(val):
    return np.log(-np.log(val))


def calculate_xs(data):
    gamble_xs = data[data[:, CHOICE_COL] == GAMBLE][:, XS_COL]
    safe_xs = data[data[:, CHOICE_COL] == SAFE][:, XS_COL]
    if len(gamble_xs) == 0:
        return np.min(safe_xs) / 2
    elif len(safe_xs) == 0:
        xg = data[0][XG_COL]
        return (np.max(gamble_xs) + xg) / 2
    else:  # מה עם מקרה של חפיפה? todo
        return (np.max(gamble_xs) + np.min(safe_xs)) / 2


def find_sigma_alpha(data):
    sigma_values, alpha_values = [], []
    for number_of_subject in range(1, NUMBER_OF_SUBJECTS + 1):
        subject_data = data[data[:, SUBJECT_COL] == number_of_subject]
        xg_list, p_list, average_xs_list = [], [], []
        for p in np.unique(subject_data[:, P_COL]):
            subject_data_xg_p_pair = subject_data[subject_data[:, P_COL] == p]  # take only the (xg, p) pairs (xg is
            # always the same for a specific p) of this subject in this trial
            p_list.append(p), xg_list.append(subject_data_xg_p_pair[0][XG_COL])
            xs = calculate_xs(subject_data_xg_p_pair)  # calculate xs for this pair (xg, p)
            average_xs_list.append(xs)
        p_list, xg_list, average_xs_list = np.array(p_list), np.array(xg_list), np.array(average_xs_list)
        x_list = np.log(-np.log(p_list))
        y_list = np.log(-np.log(average_xs_list / xg_list))
        # calculate sigma, alpha of current subject, current trial, current (xg, p) pair
        poly = np.polyfit(x_list, y_list, LINEAR)
        alpha, sigma = poly[0], poly[1]
        sigma_values.append(sigma)
        alpha_values.append(alpha)
    return alpha_values, (1 / np.exp(sigma_values))


def show_graph_comparing_trials(x, y, parameter_to_plot):
    plt.figure()
    plt.title(f"{parameter_to_plot} in second trial as a function of {parameter_to_plot} in the first trial")
    plt.scatter(x, y, color="blue", s=8)
    plt.xlabel(f"{parameter_to_plot} first trial")
    plt.ylabel(f"{parameter_to_plot} second trial")
    plt.plot(x, x, label="y=x", color="green")
    plt.grid()
    plt.show()


def main_1():
    list_of_xs = [question1_round(1)]
    for i in range(6):
        cur_xs = question1_round(list_of_xs[i])
        list_of_xs.append(cur_xs)
    print(list_of_xs)
    question1_show_graph(list_of_xs, [0.5 ** i for i in range(1, 8)])
    show_utility_function_for_different_sigmas()
    show_pi_function_for_different_alphas()


def main_2():
    data_matrix = np.loadtxt("ex10_q2_data.csv", delimiter=",", skiprows=1)
    data_matrix[:, 2] = data_matrix[:, 2] / 100  # convert percentage into probability in range [0,1]
    first_trial_data = data_matrix[data_matrix[:, NUMBER_OF_TRIAL_COL] == 1]
    sec_trial_data = data_matrix[data_matrix[:, NUMBER_OF_TRIAL_COL] == 2]
    alpha_1, sigma_1 = find_sigma_alpha(first_trial_data)
    alpha_2, sigma_2 = find_sigma_alpha(sec_trial_data)
    show_graph_comparing_trials(alpha_1, alpha_2, "alpha")
    show_graph_comparing_trials(sigma_1, sigma_2, "sigma")
    print(np.mean(alpha_1), np.mean(sigma_1))


main_1()
main_2()
