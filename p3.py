import numpy as np

inputs = [1, 2, 3, 2.5]
# The weights matrix implies we have
# three neurons
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# Order of weights and inputs matters because
# weights is a matrix and inputs is a vector
output = np.dot(weights, inputs) + biases
print(output)
