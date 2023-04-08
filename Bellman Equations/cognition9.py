import matplotlib.pyplot as plt
import random
import math

DISCOUNTING_PARAMETER = 0.5


# part 1
def value_iteration():
    # initialize the states
    next_v_home_state, next_v_out_state = 0, 0
    epsilon = math.pow(10, -10)
    v_home_state, v_out_state = 0, 0
    num_of_iterations = 0
    z = 0.5
    flag = True
    while flag or (epsilon < max(abs(next_v_home_state - v_home_state), abs(next_v_out_state - v_out_state)) and \
                   num_of_iterations < 5000):
        # updating the states values
        flag = False
        v_home_state = next_v_home_state
        v_out_state = next_v_out_state

        home_reward_switch = 1 + 0.4 * v_out_state + 0.1 * v_home_state
        home_reward_stay = 0.5 * v_home_state
        next_v_home_state = max(home_reward_stay, home_reward_switch)

        out_reward_switch = 0.5 * v_home_state
        out_reward_stay = 2 + 0.5 * v_out_state
        next_v_out_state = max(out_reward_stay, out_reward_switch)

        num_of_iterations += 1
    return v_home_state, v_out_state


# part 2, question 1
def find_reward_and_next_state(cur_state, action):
    if cur_state == "home" and action == "stay":
        return "home", 0
    elif cur_state == "home":  # action == "switch":
        next_state = random.choices(["home", "out"], [0.2, 0.8])[0]
        return next_state, 1
    elif action == "stay":  # cur_state == "out"
        return "out", 2
    else:  # cur_state == "out", action == "switch
        return "home", 0


# part 2, question 2. find the values of the states for some policy (in this case check for policy that gives
# 0.5 chances to choose each action in each state)
def TD_learning_algorithm():
    v_home_state = 0
    v_out_state = 0
    learning_rate = 0.01
    number_of_updates = 3000
    list_of_home_values = [0]
    list_of_out_values = [0]
    current_state = random.choices(["home", "out"], [0.5, 0.5])[0]
    # policy = 0.5 chances to choose each action in each state
    for i in range(number_of_updates):
        action = random.choices(["stay", "switch"], [0.5, 0.5])[0]
        # home update
        next_state, cur_reward = find_reward_and_next_state(current_state, action)
        if next_state == "home":
            next_state_value = v_home_state
        else:
            next_state_value = v_out_state
        # updating the values
        if current_state == "home":
            v_home_state += learning_rate * (cur_reward + DISCOUNTING_PARAMETER * next_state_value - v_home_state)
        else:
            v_out_state += learning_rate * (cur_reward + DISCOUNTING_PARAMETER * next_state_value - v_out_state)

        list_of_home_values.append(v_home_state)
        list_of_out_values.append(v_out_state)
        current_state = next_state

    return list_of_home_values, list_of_out_values


# part 2, question 2 + 3
def show_values_progress_in_graph(home_values, out_values, res_home, res_out, i):
    x = [i for i in range(0, len(home_values))]
    y_out = [res_home] * len(out_values)
    y_home = [res_out] * len(home_values)
    plt.title(f"question {i}")
    plt.plot(x, home_values, label="home value", color="blue")
    plt.plot(x, out_values, label="out value", color="purple")
    plt.plot(x, y_out, '--', label="out value result in question 1", color="red")
    plt.plot(x, y_home, '--', label="home value result in question 1", color="orange")
    plt.legend()
    plt.grid()
    plt.show()


# part 2 , question 3
def Q_learning_algorithm():
    Q_home_stay, Q_home_switch, Q_out_stay, Q_out_switch = 0, 0, 0, 0
    learning_rate = 0.01
    number_of_updates = 5000
    out_value_list = [0]
    home_value_list = [0]
    current_state = random.choices(["home", "out"], [0.5, 0.5])[0]
    for i in range(number_of_updates):
        action = random.choices(["stay", "switch"], [0.5, 0.5])[0]
        next_state, reward = find_reward_and_next_state(current_state, action)
        if next_state == "out":
            Q_max = max(Q_out_switch, Q_out_stay)
        else:
            Q_max = max(Q_home_stay, Q_home_switch)

        if current_state == "home":
            if action == "stay":
                Q_home_stay += learning_rate * (reward + (DISCOUNTING_PARAMETER * Q_max) - Q_home_stay)
            else:
                Q_home_switch += learning_rate * (reward + (DISCOUNTING_PARAMETER * Q_max) - Q_home_switch)

        else:  # out == current_state
            if action == "stay":
                Q_out_stay += learning_rate * (reward + (DISCOUNTING_PARAMETER * Q_max) - Q_out_stay)
            else:
                Q_out_switch += learning_rate * (reward + (DISCOUNTING_PARAMETER * Q_max) - Q_out_switch)

        home_value_list.append(max(Q_home_switch, Q_home_stay))
        out_value_list.append(max(Q_out_switch, Q_out_stay))

        current_state = next_state

    return home_value_list, out_value_list


def main():
    print(value_iteration())
    home_values_TD, out_values_TD = TD_learning_algorithm()
    show_values_progress_in_graph(home_values_TD, out_values_TD, 33 / 19, 161 / 133, 2)
    home_values_Q, out_values_Q = Q_learning_algorithm()
    show_values_progress_in_graph(home_values_Q, out_values_Q, 26 / 9, 4, 3)


main()
