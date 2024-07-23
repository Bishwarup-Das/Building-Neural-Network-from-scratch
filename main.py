# Building neural network from scratch

# import required dependencies
import numpy as np
import matplotlib.pyplot as plt

# Defining function to initialize weights and bias for every neuron according to the dimension size of each layer
def init_params(layer_dims):
    np.random.seed(3)
    params = {}
    dims_size = len(layer_dims)
    for i in range(1, dims_size):
        params['W'+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])*0.01
        params['b'+str(i)] = np.zeros((layer_dims[i], 1))
    return params

# Defining the sigmoid function to compute the value for given Z
# Where z (linear hypothesis) = W(weight matrix)*X(input)+b(bias vector)
def sigmoid(z):
    sigmoid_value = 1/(1+np.exp(-z))
    cache = z
    return sigmoid_value, cache




# Defining the forward propagation function which will calculate value for one layer and feed that output to the next layer and so on...
def forward_prop(X, params):
    layers_input = X.T         # Input data for first layer
    caches = []
    params_len = len(params) // 2 + 1
    for i in range(1, params_len):
        prev_layers_input = layers_input
        # Linear Hypothesis
        z = np.dot(params['W'+str(i)], prev_layers_input) + params['b'+str(i)]
        # Storing linear cache
        linear_cache = (prev_layers_input, params['W'+str(i)], params['b'+str(i)])
        # Applying sigmoid on linear hypothesis
        layers_input, activation_cache = sigmoid(z)
        # storing the both linear and activation cache
        cache = (linear_cache, activation_cache)
        caches.append(cache)
    return layers_input, caches




# Defining the loss function to find the error difference between the actual and the predicted value
def cost_function(A, Y):
    m = Y.shape[1]
    cost = (-1/m)*np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    cost = np.squeeze(cost)
    return cost

# Defining one step backward function to get the gradient values for Sigmoid function of one layer
def one_step_backward(dA, cache):
    linear_cache, activation_cache = cache
    Z = activation_cache
    A = 1/(1+np.exp(-Z))
    dZ = dA * A * (1 - A)   # calculating the derivative of sigmoid function
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

# Defining backpropagation function to get gradients for all layers
def backprop(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = one_step_backward(dAL, current_cache)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = one_step_backward(grads["dA" + str(l + 2)], current_cache)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

# Defining function to update parameters using gradient descent
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
    return parameters

# Defining function to train the neural network
def train(X, Y, layer_dims, epochs, lr):
    params = init_params(layer_dims)
    cost_history = []
    for i in range(epochs):
        Y_hat, caches = forward_prop(X, params)
        cost = cost_function(Y_hat, Y)
        cost_history.append(cost)
        grads = backprop(Y_hat, Y, caches)
        params = update_parameters(params, grads, lr)
        if i % 100 == 0:
            print(f"Cost after epoch {i}: {cost}")
    return params, cost_history



# Example usage
if __name__ == "__main__":
    # Generate a larger dataset
    X = np.random.rand(1000, 10)  # 1000 samples with 10 features each
    Y = np.random.randint(0, 2, (1, 1000))  # Binary labels for 1000 samples
    layer_dims = [10, 5, 1]  # Adjusted according to the new data dimensions

    params, cost_history = train(X, Y, layer_dims, epochs=1000, lr=0.01)

    plt.plot(cost_history)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Cost Function')
    plt.show()








