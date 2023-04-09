from PCA import *


def main():
    mat_first = np.loadtxt("wines.csv", delimiter=",", skiprows=1)
    wines_number = mat_first.transpose()[0]
    create_graph(mat_first)
    mat = preprocessing(mat_first)
    vals, vectors = calculate_eigenvalues_eigenvectors_of_c(mat)  # for the vectors, each row is a vector
    show_cumulative_variance(vals)
    U = vectors[:2].transpose()  # take the first two rows (first two vectors) and present them in a way each
    # col is a vec
    reduce_dimension_and_plot(mat, U, wines_number)  # (178 X 13) , (13 X 2)
    U_other = np.array([vectors[0], vectors[12]]).transpose()  # take the first and last rows and present them in a
    # way each col is a vec
    reduce_dimension_and_plot(mat, U_other, wines_number)


if __name__ == "__main__":
    main()
