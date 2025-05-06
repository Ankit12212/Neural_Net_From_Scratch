import numpy as np
import nnfs
from nnfs.datasets import spiral_data

class dense_layer:
    def __init__(self, input_no, neuron_no):
        self.weights= 0.05*np.random.randn(input_no, neuron_no)
        self.biases = np.zeros((1, neuron_no))

    def forward(self, inputs):
        self.output = np.dot(inputs , self.weights) + self.biases

class ReLU_Activation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class softmax_Activation:
    def forward(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        self.output = exp_inputs / np.sum(exp_inputs, axis=1 , keepdims=True)

class loss_categoricalClassEntropy:
    def forward(self, y_pred, y_target):

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        target_length = len(y_target.shape)

        if target_length == 1:
            correct_confidence = [y_pred_clipped[range(len(y_pred)), y_target]]

        elif target_length == 2:
            correct_confidence = np.sum((y_pred_clipped * y_target), axis = 1)

        self.sample_loss = -np.log(correct_confidence)

    def data_loss(self):
        return np.mean(self.sample_loss)

X, y = spiral_data(samples=100, classes=3)

#for layer 1 
layer1 = dense_layer(2,3)
act1 = ReLU_Activation()
layer1.forward(X)
act1.forward(layer1.output)

#for layerr 2
layer2 = dense_layer(3,3)
act2 = softmax_Activation()
layer2.forward(act1.output)
act2.forward(layer2.output)

#predicted confidence score 
print(act2.output[:5])

#loss calculation
loss1 = loss_categoricalClassEntropy()
loss1.forward(act2.output, y)
print(f'Loss: {loss1.data_loss()}')
