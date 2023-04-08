import numpy as np
import matplotlib.pyplot as plt
import random

from utils import loadMNISTLabels, loadMNISTImages
from ff import FF

NUMBER_OF_PHOTOS = 10000
NUM_OF_EXAMPLES = 10
## Loading the dataset

y_test = loadMNISTLabels('../MNIST_data/t10k-labels-idx1-ubyte')
y_train = loadMNISTLabels('../MNIST_data/train-labels-idx1-ubyte')

X_test = loadMNISTImages('../MNIST_data/t10k-images-idx3-ubyte')  # 784 X 10000 נעשה טרנספוז כי כל עמודה מייצגת תמונה
X_train = loadMNISTImages('../MNIST_data/train-images-idx3-ubyte')

## random permutation of the input
# uncomment this to use a fixed random permutation of the images


perm = np.random.permutation(784)
X_test = X_test[perm, :]
X_train = X_train[perm, :]

## Parameters
layers_sizes = [784, 30, 10]  # flexible, but should be [784,...,10]
epochs = 10
eta = 0.1
batch_size = 20

## Training
net = FF(layers_sizes)
steps, test_acc = net.sgd(X_train, y_train, epochs, eta, batch_size, X_test, y_test)

## plotting learning curve and visualizing some examples from test set

plt.figure()
plt.plot(steps, test_acc, color="firebrick", linewidth=1)
plt.xlabel("steps")
plt.ylabel("test accuracy")
plt.grid()
plt.title("part 2 question 1")
plt.show()

# next part
y_test_t = np.transpose(y_test)
x_test_t = np.transpose(X_test)
list_of_examples = []
list_of_predictions = []
for i in range(10):
    for j in range(NUM_OF_EXAMPLES):
        row = random.randint(0, NUMBER_OF_PHOTOS - 1)
        while np.argmax(y_test_t[row]) != i:
            row = random.randint(0, NUMBER_OF_PHOTOS - 1)
        list_of_examples.append(x_test_t[row])
        network_output = net.predict(x_test_t[row])
        # make it a vector represent a digit
        list_of_predictions.append(np.argmax(network_output))

fig = plt.figure(figsize=(10, 10))
plt.axis('off')
plt.title("question 2 part 2 with mix")
rows, columns = 10, 10
count = 1
for i in range(10):
    for j in range(10):
        fig.add_subplot(rows, columns, count)
        image = np.reshape(list_of_examples[i * 10 + j], (28, 28))
        plt.axis('off')
        if list_of_predictions[i * 10 + j] == i:
            plt.imshow(image)
        else:
            plt.imshow(image, cmap='hot')
        count += 1
plt.show()
