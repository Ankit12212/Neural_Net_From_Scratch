# for this section you need to install the nnfs package developed especially for this course. It provides:
# - Datasets (like spiral or vertical datasets)
# - Helper functions (like one-hot encoding)
# - Some layers and activation functions for building basic neural networks manually (without using TensorFlow or PyTorch).

# The nnfs.init() does three things: it sets the random seed to 0 (by the default), creates a float32​ dtype default, and overrides the original dot product from NumPy. 

import numpy as np 
import nnfs
from nnfs.datasets import spiral_data

class layer_dense:
    def __init__(self, neurons_no, inputs_no):
        self.weights = 0.001*np.random.randn(inputs_no, neurons_no)
        self.bias = np.zeros((1, neurons_no))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias

#creates a non-linear spiral dataset
x, y = spiral_data(100, 3)   

#creates a layer with 2 inputs and 3 neruons
layer1 = layer_dense(3,2)

layer1.forward(x)

print(layer1.output[:200])
    