from PCA import *


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


if __name__ == "__main__":
    main()
