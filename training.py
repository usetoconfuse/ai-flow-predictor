import numpy as np
import pandas as pd

from data_processing import destandardise_val


# INITIALISE A NEURAL NETWORK

# Generate weights of the neural network
# size: number of nodes on this layer
# inputs: number of nodes on previous layer
def weighted_layer(size, inputs):
    lim = 2/inputs
    gen = np.random.default_rng()
    return gen.uniform(low=-lim, high=lim, size=(size, inputs+1))


# Generate weights for a neural network with a number of hidden layers
# Returns a 3D array of layers, nodes, and weights
# e.g., network[1][2][4] refers to layer 2, node 3, weight 5
# The first layer network[0] is the input layer which has no weights
# the first weight for each node (network[layer][node][0]) is the bias
def initialise_network(inputs, hidden_layers, layer_size):
    # First hidden layer
    network = [weighted_layer(layer_size, inputs)]
    # Rest of hidden layers
    for i in range(hidden_layers-1):
        network.append(weighted_layer(layer_size, layer_size))
    # Output layer (1 node)
    network.append(weighted_layer(1, layer_size))

    return network




# TRAIN NETWORK


def sigmoid(x):
    return 1 / (1+pow(np.e, -x))


def forward_pass(network, row):
    inputs = row[:-1]
    u_vals = []

    for l in range(len(network)):
        layer = network[l]
        outputs = []
        for node in layer:
            # Bias plus weighted inputs
            node_val = node[0] + np.sum(np.multiply(inputs, node[1:]))

            # Activation function for hidden layers
            if l != len(network)-1:
                node_val = sigmoid(node_val)

            outputs.append(node_val)
        inputs = outputs
        u_vals.append(outputs)

    return u_vals

# For each row of data, calculate the output value u at each node
# from the values at the nodes in the previous layer
def backpropagate(network, row, lrn):

    predictand = row[-1]

    # FORWARD PASS
    u_vals = forward_pass(network, row)


    # BACKWARD PASS

    # Calculate delta_o and squared error
    output = u_vals[-1][0]
    error = predictand - output
    delta_output = error * output * (1-output)

    deltas = [[delta_output]]
    # Iterate through hidden layers in reverse to get deltas
    for i in range(len(u_vals)-2, -1, -1):
        # Calculate delta_o * f'(S_x) for every node x in the layer
        layer_deltas = [
            delta_output * x * (1-x)
            for x in u_vals[i]
        ]
        # Then multiply each of these values by w_x,o to get delta_x for every x
        layer_deltas = np.multiply(layer_deltas, network[i+1][0][1:])

        # Prepend to deltas list as we are iterating backwards
        deltas.insert(0, layer_deltas)

    # Prepend input node values for weight adjustment calculations
    u_vals.insert(0, row[:-1])

    # Update weights in the network
    new_network = []
    for l in range(len(network)):
        layer = network[l]
        new_layer = []
        for n in range(len(layer)):
            node = layer[n]
            new_node = [node[0] + lrn * deltas[l][n]]
            new_node = (new_node +
                        [node[i] + lrn * deltas[l][n] * u_vals[l][n]
                         for i in range(1, len(node))])
            new_layer.append(new_node)
        new_network.append(new_layer)

    return new_network, output


# Backpropagation: forward pass and backward pass over every row
# Return trained network and root-mean-square error over the data
def train(network, data, lrn):
    output_arr = []
    for row in data:
        network, output = backpropagate(network, row, lrn)
        output_arr.append(output)
    return network, pd.DataFrame(output_arr)


# Forward pass that returns modelled and predicted values for visualisation
def predict(network, data):
    modelled_arr = []
    for row in data:
        output = forward_pass(network, row)[-1][0]
        modelled_arr.append(output)
    return pd.DataFrame(modelled_arr)