import numpy as np


class FF(object):
    """A simple FeedForward neural network"""

    def __init__(self, layerDims):
        super(FF, self).__init__()
        n_weights = len(layerDims) - 1
        self.weights = []
        for i in range(n_weights):
            self.weights.append(0.1 * np.random.randn(layerDims[i + 1], layerDims[i]))

    def sgd(self, X, y, epochs, eta, mb_size, Xtest, ytest):
        N = X.shape[1]
        n_mbs = int(np.ceil(N / mb_size))
        acc = self.eval_test(Xtest, ytest)

        updates = 0
        steps = [updates]
        test_acc = [acc]
        print("Starting training, test accuracy: {0}".format(acc))

        for i in range(epochs):
            perm = np.random.permutation(N);
            for j in range(n_mbs):
                X_mb = X[:, perm[j * mb_size:(j + 1) * mb_size]]
                y_mb = y[:, perm[j * mb_size:(j + 1) * mb_size]]

                grads = self.backprop(X_mb, y_mb)

                for k, grad in enumerate(grads):
                    self.weights[k] = self.weights[k] - (eta / mb_size) * grad

                updates = updates + 1
                if updates % 50 == 0:
                    steps.append(updates)
                    test_acc.append(self.eval_test(Xtest, ytest))

            acc = self.eval_test(Xtest, ytest)
            print("Done epoch {0}, test accuracy: {1}".format(i + 1, acc))

        steps = np.asarray(steps)
        steps = steps / n_mbs

        return steps, test_acc

    def backprop(self, X, y):

        # X is a matrix of size input_dim*mb_size
        # y is a matrix of size output_dim*mb_size
        # you should return a list 'grads' of length(weights) such
        # that grads[i] is a matrix containing the gradients of the
        # loss with respect to weights[i].

        # ForwardPass
        S0 = X  # now input layer is a matrix
        H1 = np.matmul(self.weights[0], S0)  # this is matrix of inputs according to the examples
        S1 = FF.activation(H1)
        H2 = np.matmul(self.weights[1], S1)
        network_output = FF.activation(H2)

        # BackwardPass
        grad_output_layer = FF.loss_deriv(network_output, y)
        deriv_H2 = FF.activation_deriv(H2)
        D2 = grad_output_layer * deriv_H2
        deriv_H1 = FF.activation_deriv(H1)
        D1 = np.matmul(np.transpose(self.weights[1]), D2) * deriv_H1

        # Gradients
        first = np.matmul(D1, np.transpose(S0))
        sec = np.matmul(D2, np.transpose(S1))

        # returns the desire output for some example x
        return [first, sec]

    def predict(self, x):
        a = x
        for w in self.weights:
            a = FF.activation(np.dot(w, a))

        return a

    def eval_test(self, Xtest, ytest):
        ypred = self.predict(Xtest)
        ypred = ypred == np.max(ypred, axis=0)

        return np.mean(np.all(ypred == ytest, axis=0))

    def activation(x):
        return np.tanh(x)

    def activation_deriv(x):
        return 1 - (np.tanh(x) ** 2)

    def loss_deriv(output, target):
        # Derivative of loss function with respect to the activations
        # in the output layer.
        # we use quadratic loss, where L=0.5*||output-target||^2
        # YOUR CODE HERE
        return np.subtract(output, target)
