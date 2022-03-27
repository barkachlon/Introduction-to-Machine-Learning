#Bar Kachlon 207630864
#sector A
import numpy as np
import matplotlib.pyplot as plt

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#Initilize variables
inputs = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
target_output = np.array([[0],[1],[1],[0],[1],[0],[0],[1]])
E = []
Iter = 2000
Set = 100
eta = 2
mu, sigma = 0, 1
in_Layer_Neurons, hid_Layer_Neurons, out_Layer_Neurons = 3,3,1
# Training
for _ in range(Set):
    # Random weights and bias
    hid_W = np.random.normal(size=(in_Layer_Neurons, hid_Layer_Neurons))
    hid_Layer_bias = np.random.normal(size=(1, hid_Layer_Neurons))
    output_W = np.random.normal(size=(hid_Layer_Neurons, out_Layer_Neurons))
    output_Layer_bias = np.random.normal(size=(1, out_Layer_Neurons))
    for _ in range(Iter):
        # Forward Propagation
        a_j_hidden_layer = np.dot(inputs, hid_W)
        a_j_hidden_layer += hid_Layer_bias
        hid_Layer_output = sigmoid(a_j_hidden_layer)
        a_k_output_layer = np.dot(hid_Layer_output, output_W)
        a_k_output_layer += output_Layer_bias
        y_estimated_output = sigmoid(a_k_output_layer)
        # Backpropagation
        delta_estimated_output = (y_estimated_output -target_output ) * sigmoid_derivative(y_estimated_output)
        delta_hid_layer = (delta_estimated_output.dot(output_W.T)) * sigmoid_derivative(hid_Layer_output)
        # Updating Weights and Bias
        output_W -= hid_Layer_output.T.dot(delta_estimated_output) * eta
        output_Layer_bias -= np.sum(delta_estimated_output, axis=0, keepdims=True) * eta
        hid_W -= inputs.T.dot(delta_hid_layer) * eta
        hid_Layer_bias -= np.sum(delta_hid_layer, axis=0, keepdims=True) * eta
    temp_E = 0.125 * ((y_estimated_output-target_output)**2).sum()
    E.append(temp_E)
plt.figure()
plt.subplot(211)
plt.plot(range(Set), E)
plt.title('Mean squared error as a function of set with 3 hidden layers')
plt.ylabel('Mean squared error Function')
#sector B

#Initilize variables
E = []
hid_Layer_Neurons = 6

# Training
for _ in range(Set):
    # Random weights and bias
    hid_W = np.random.normal(size=(in_Layer_Neurons, hid_Layer_Neurons))
    hid_Layer_bias = np.random.normal(size=(1, hid_Layer_Neurons))
    output_W = np.random.normal(size=(hid_Layer_Neurons, out_Layer_Neurons))
    output_Layer_bias = np.random.normal(size=(1, out_Layer_Neurons))
    for _ in range(Iter):
        # Forward Propagation
        a_j_hidden_layer = np.dot(inputs, hid_W)
        a_j_hidden_layer += hid_Layer_bias
        hid_Layer_output = sigmoid(a_j_hidden_layer)
        a_k_output_layer = np.dot(hid_Layer_output, output_W)
        a_k_output_layer += output_Layer_bias
        y_estimated_output = sigmoid(a_k_output_layer)
        # Backpropagation
        delta_estimated_output = ( y_estimated_output-target_output ) * sigmoid_derivative(y_estimated_output)
        delta_hid_layer = (delta_estimated_output.dot(output_W.T)) * sigmoid_derivative(hid_Layer_output)
        # Updating Weights and Bias
        output_W -= hid_Layer_output.T.dot(delta_estimated_output) * eta
        output_Layer_bias -= np.sum(delta_estimated_output, axis=0, keepdims=True) * eta
        hid_W -= inputs.T.dot(delta_hid_layer) * eta
        hid_Layer_bias -= np.sum(delta_hid_layer, axis=0, keepdims=True) * eta
    temp_E = 0.125 * ((y_estimated_output-target_output)**2).sum()
    E.append(temp_E)
plt.subplot(212)
plt.plot(range(Set), E)
plt.title('Mean squared error  as a function of set with 6 hidden layers')
plt.ylabel('Mean squared error Function')
plt.xlabel('set')
plt.show()
