import numpy as np
import pandas as pd


# =========================== INITIALISE A NEURAL NETWORK =======================================

# Generate weights of the neural network
# size: number of nodes on this layer
# inputs: number of nodes on previous layer
def weighted_layer(size, inputs):
    lim = 2 / inputs
    gen = np.random.default_rng()
    return gen.uniform(low=-lim, high=lim, size=(size, inputs + 1))


# Generate weights for a neural network with a number of hidden layers
# Returns a 3D array of layers, nodes, and weights
# e.g., network[1][2][4] refers to layer 2, node 3, weight 5
# The first layer network[0] is the input layer which has no weights
# the first weight for each node (network[layer][node][0]) is the bias
def initialise_network(inputs, hidden_layers, layer_size):
    # First hidden layer
    network = [weighted_layer(layer_size, inputs)]
    # Rest of hidden layers
    for i in range(hidden_layers - 1):
        network.append(weighted_layer(layer_size, layer_size))
    # Output layer (1 node)
    network.append(weighted_layer(1, layer_size))

    return network



# =============================== TRAIN NETWORK =======================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def forward_pass(network, row):
    inputs = row[:-1]
    u_vals = []

    # For each row of data, calculate the output value u at each node in a layer
    # from the values at the nodes in the previous layer
    for layer in network:
        outputs = []
        for node in layer:
            # Bias plus weighted inputs
            node_val = node[0] + np.dot(inputs, node[1:])

            # Activation function for hidden/output layers
            node_val = sigmoid(node_val)

            outputs.append(node_val)
        inputs = outputs
        u_vals.append(outputs)

    return u_vals



def backpropagate(network, row, lrn):
    predictand = row[-1]

    # FORWARD PASS
    u_vals = forward_pass(network, row)


    # BACKWARD PASS

    # Calculate delta_o and squared error
    output = u_vals[-1][0]
    error = predictand - output
    delta_output = error * sigmoid_derivative(output)

    deltas = [[delta_output]]
    # Iterate through hidden layers in reverse to get deltas
    for layer in range(len(u_vals) - 2, -1, -1):
        # Calculate delta_o * f'(S_x) for every node x in the layer
        layer_deltas = [
            delta_output * sigmoid_derivative(x)
            for x in u_vals[layer]
        ]
        # Then multiply each of these values by w_x,o to get delta_x for every node x in the layer
        layer_deltas = np.multiply(layer_deltas, network[layer + 1][0][1:])

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

            # For each node in each layer, calculate the adjusted weights
            node = layer[n]
            node_delta = deltas[l][n]
            new_node = [node[0] + lrn * deltas[l][n]]
            for weight in range(1, len(node)):

                # For each input weight on this node from a node in the previous layer,
                # get the output value u_val of the node in the previous layer
                # and use this to build the new set of weights for this node
                u_val = u_vals[l][weight-1]
                new_node = new_node + [(node[weight] + lrn * node_delta * u_val)]
            new_layer.append(new_node)
        new_network.append(new_layer)

    return new_network


# Backpropagation: forward pass and backward pass over every row
# Train a network on a given DataFrame
def train(network, data_frame, lrn, epochs):

    data_arr = data_frame.to_numpy()

    for epoch in range (1, epochs+1):
        for row in data_arr:
            network = backpropagate(network, row, lrn)
        if (epoch % 100) == 0:
            print(f"{epoch} epochs done")

    return network


# Get predicted values from a network over a DataFrame
def predict(network, data_frame):
    data_arr = data_frame.to_numpy()
    pred_arr = []
    for row in data_arr:
        pred_arr.append(forward_pass(network, row)[-1][0])
    return pred_arr


if __name__ == "__main__":
    data = pd.DataFrame([[1, 0, 1]])

    test_network = [[[1, 3, 4], [-6, 6, 5]], [[-3.92, 2, 4]]]
    test_network = train(test_network, data, 0.1, 10000)

    print(predict(test_network, data))