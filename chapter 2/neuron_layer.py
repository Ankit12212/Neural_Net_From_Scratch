import numpy as np

inputs = [1, 2, 3, 2.5] 
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

outputs = []

for neuron, bias in zip(weights, biases):
    output = 0
    for input, weight in zip(inputs, neuron):
        output += input * weight
    output += bias
    outputs.append(output)

# outputs = np.dot(inputs, np.array(weights).T) + biases

print(outputs)