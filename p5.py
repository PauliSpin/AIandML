import numpy as np
import nnfs  # See below for details of nnfs
# https://www.youtube.com/watch?v=gmjzbpSVY1A
from nnfs.datasets import spiral_data

nnfs.init()


# X Feature Set
# y target or classification
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)  # 100 Feature sets of 3 classes


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# layer1 = Layer_Dense(4, 5)  # 4 is the Number of features per sample
layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)

print(layer1.output)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# NNFS
# ----
#
# pip install nnfs
# nnfs allows you to get the code from that is being used in the tutorial:
# Inputs are X; our inout data to the neural network
# (venv) PS C:\Users\rchotalia\Documents\VisualStudioCode\AIandML> pip install nnfs
# Collecting nnfs
#   Downloading nnfs-0.5.0-py3-none-any.whl (9.1 kB)
# Requirement already satisfied: numpy in c:\users\rchotalia\documents\visualstudiocode\aiandml\venv\lib\site-packages (from nnfs) (1.18.4)
# Installing collected packages: nnfs
# Successfully installed nnfs-0.5.0
# (venv) PS C:\Users\rchotalia\Documents\VisualStudioCode\AIandML> nnfs

# Neural Networks from Scratch in Python Tool.

# Basic usage:
# nnfs command [parameter1 [parameter2]]

# Detailed usage:
# nnfs info | code video_part [destination]

# Commands:
#   info    Prints information about the book
#   code    Creates a file containing the code of given video part
#           in given location. Location is optional, example:
#           nnfs code 2 nnfs/p02.py
#           will create a file p02.py in a nnfs folder containing
#           the code of part 2 of video tutorial

# (venv) PS C:\Users\rchotalia\Documents\VisualStudioCode\AIandML>
