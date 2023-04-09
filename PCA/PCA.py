import numpy as np
import matplotlib.pyplot as plt


def create_graph(mat):
    """ This function show
    Args:
        mat:
    """
    x = mat.transpose()[1]
    y = mat.transpose()[2]
    color = mat.transpose()[0].astype(int)
    plt.title("purple for wine 1, green for wine 2, yellow for wine 3")
    plt.scatter(x, y, c=color)
    plt.grid()
    plt.show()


def preprocessing(mat):
    """ This function preprocessing a matrix by normalized its entries (reduce the mean and divide by the std)

        Args:
            mat: a matrix to preprocess
    """
    mat = (mat.transpose()[1:]).transpose()  # get rid of first column with info about the wines number
    average_vec = mat.mean(0)  # 1 X 13 vector of average of each attribute (0 means average by cols)
    mat = mat - average_vec
    standard_deviation = np.std(mat, axis=0)
    mat = mat / standard_deviation
    return mat


def calculate_eigenvalues_eigenvectors_of_c(mat):
    """ This function finds the eigenvalues of the given matrix and a corresponding eigenvectors
        Args:
            mat (NXM ndarray): the matrix to find the eigenvalues and eigenvectors for

        Returns:
            sorted_eigenvalues: the eigenvalues in descending order
            sorted_eigenvectors: M eigenvectors corresponding to the sorted_eigenvalues
    """
    mat_rows_are_properties = np.array(mat.transpose())
    mat_rows_are_examples = np.array(mat)
    correlation_mat = np.matmul(mat_rows_are_properties, mat_rows_are_examples) / mat.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(correlation_mat)
    eigenvectors = eigenvectors.transpose()  # each row is a vector
    eigenvalues_idx = eigenvalues.argsort()
    sorted_eigenvalues = eigenvalues[eigenvalues_idx[::-1]]
    sorted_eigenvectors = eigenvectors[eigenvalues_idx[::-1]]
    return sorted_eigenvalues, sorted_eigenvectors


def show_cumulative_variance(eigenvalues):
    """ This function show the explained variance for different (increasing) size of eigenvalues selection
        Args:
            eigenvalues: all the eigenvalues of some matrix
    """
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
    """ This function

    Args:
        mat:
        U:
        wines_number:
    """
    result = np.matmul(mat, U)  # 178 X 2, each row is the examples after the dimensions' reduction, y
    plt.figure()
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    plt.title("purple for wine 1, green for wine 2, yellow for wine 3")
    plt.scatter(result.transpose()[0], result.transpose()[1], c=wines_number)
    plt.grid()
    plt.show()
