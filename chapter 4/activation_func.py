import numpy as np 
import nnfs 
from nnfs.datasets import spiral_data

class dense_layer:
    def __init__(self, inputs_no, neurons_no):
        self.weights = 0.001*np.random.randn(inputs_no, neurons_no)
        self.biases = np.zeros((1, neurons_no))

    def forward(self, input_values):
        self.output = np.dot(input_values, self.weights) + self.biases
        
#ReLU activation function, for hidden layers 
class relu_activation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

#Softmax activation function, for output layer
class softmax_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs- np.max(inputs, axis = 1, keepdims=True))
        probablities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = probablities

x, y = spiral_data(samples=100, classes=3)


#for first layer 
layer1 = dense_layer(2,5) #inputs_no = 2 because spiral data is passed to it, and each batch is a 2D data point i.e x, & y coordinate
act1 = relu_activation()

#for second layer 
layer2 = dense_layer(5,3) #number of neruons in last layer matches the number of classes of input 
act2 = softmax_softmax()

#spiral_data --> layer1 --> act1 --> layer2 --> act2

#operation in layer 1
layer1.forward(x)
act1.forward(layer1.output)

#operation in layer 2 
layer2.forward(act1.output)
act2.forward(layer2.output)

print(act2.output[:5])