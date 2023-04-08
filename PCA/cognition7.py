import numpy as np
import matplotlib.pyplot as plt


# question 1
def create_matrix():
    mat = np.loadtxt("wines.csv", delimiter=",", skiprows=1)
    return mat


# question 2
def create_graph(mat):
    x = mat.transpose()[1]
    y = mat.transpose()[2]
    color = mat.transpose()[0].astype(int)
    plt.title("question 2.2 - purple for wine 1, green for wine 2, yellow for wine 3")
    plt.scatter(x, y, c=color)
    plt.grid()
    plt.show()


# question 3
def preprocessing(mat):
    mat = (mat.transpose()[1:]).transpose()  # get rid of first column with info about the wines number
    average_vec = mat.mean(0)  # 1 X 13 vector of average of each attribute (0 means average by cols)
    mat = mat - average_vec
    standard_deviation = np.std(mat, axis=0)
    mat = mat / standard_deviation
    return mat


# question 4
def calculate_eigenvalues_eigenvectors_of_c(mat):  # get matrix of 179X14
    mat_rows_are_properties = np.array(mat.transpose())
    mat_rows_are_examples = np.array(mat)
    correlation_mat = np.matmul(mat_rows_are_properties, mat_rows_are_examples) / 178
    eigenvalues, eigenvectors = np.linalg.eig(correlation_mat)
    eigenvectors = eigenvectors.transpose()  # each row is a vector
    eigenvalues_idx = eigenvalues.argsort()
    sorted_eigenvalues = eigenvalues[eigenvalues_idx[::-1]]
    sorted_eigenvectors = eigenvectors[eigenvalues_idx[::-1]]
    return sorted_eigenvalues, sorted_eigenvectors


# question 5
def show_cumulative_variance(eigenvalues):
    # getting cumulative variance
    sum = np.sum(eigenvalues)
    cur_sum = 0
    list = [0]
    for i in range(13):
        cur_sum += eigenvalues[i]
        list.append(cur_sum / sum)
    # show graph of explained variance
    plt.figure()
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    plt.xlim(0, 14)
    plt.ylim(0, 1.02)
    plt.title("question 2.4 - cumulative variance")
    plt.xlabel("# of eigenvectors")
    plt.ylabel("explained variance")
    plt.scatter([i for i in range(14)], list, label="cumulative variance", color="blue", s=7)
    plt.grid()
    plt.show()


def reduce_dimension_and_plot(mat, U, wines_number):
    result = np.matmul(mat, U)  # 178 X 2, each row is the examples after the dimensions reduction, y
    plt.figure()
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    plt.title("purple for wine 1, green for wine 2, yellow for wine 3")
    plt.scatter(result.transpose()[0], result.transpose()[1], c=wines_number)
    plt.grid()
    plt.show()


def main():
    mat_first = create_matrix()
    wines_number = mat_first.transpose()[0]
    create_graph(mat_first)
    mat = preprocessing(mat_first)
    vals, vecs = calculate_eigenvalues_eigenvectors_of_c(mat)  # for the vecs, each row is a vector
    show_cumulative_variance(vals)
    U = vecs[:2].transpose()  # take the first two rows (first two vectors) and present them in a way each
    # col is a vec
    reduce_dimension_and_plot(mat, U, wines_number)  # (178 X 13) , (13 X 2)
    U_other = np.array([vecs[0], vecs[12]]).transpose()  # take the first and last rows and present them in a way each
    # col is a vec
    reduce_dimension_and_plot(mat, U_other, wines_number)


main()
