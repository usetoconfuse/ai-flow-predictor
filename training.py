import numpy as np
import data_processing as dp


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





##################################################################################################################
####################################### BACKPROPAGATION FUNCTIONS ################################################
##################################################################################################################

# Forward pass of a network over a given row of data (last value assumed to be predictand)
def forward_pass(network, row):
    inputs = row[:-1]
    u_vals = []

    # For each node in each layer, calculate u - the output of the activation function applied to the weighted sum
    # Calculate this from the values at the nodes in the previous layer
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


# Backpropagation: forward pass and backward pass over every row
def backpropagate(network, row, lrn):
    predictand = row[-1]

    # FORWARD PASS

    # Get the weighted sum at each node with the activation function applied
    u_vals = forward_pass(network, row)


    # BACKWARD PASS

    # Calculate delta_output
    output = u_vals[-1][0]
    error = predictand - output
    delta_output = error * sigmoid_derivative(output)

    # Iterate through hidden layers in reverse to get deltas at each node
    deltas = [[delta_output]]
    for layer in range(len(u_vals) - 2, -1, -1):

        # Calculate delta_output * f'(S_x) for every node x in the layer
        layer_deltas = [
            delta_output * sigmoid_derivative(u)
            for u in u_vals[layer]
        ]

        # Then multiply each of these values by w_x,output to get delta_x for every node x in the layer
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

                # For each input weight on this node from an input node in the previous layer,
                # get the value input_u_val of the input node (weighted sum with activation function applied)
                # and use this to build the new set of weights for this node
                input_u_val = u_vals[l][weight-1]
                new_node = new_node + [(node[weight] + lrn * node_delta * input_u_val)]
            new_layer.append(new_node)
        new_network.append(new_layer)

    # Return the updated network
    return new_network


##################################################################################################################
##################################################################################################################
##################################################################################################################



# ====================== ERROR CALCULATION =======================================

# Calculate root-mean-square error of a given set of predicted and actual values
def rmse_calc(predicted_vals, real_vals):
    total_sq_err = 0

    for row in range(len(predicted_vals)):
        error = real_vals[row] - predicted_vals[row]
        total_sq_err += error**2

    mse = total_sq_err / len(predicted_vals)
    return np.sqrt(mse)



# ====================== TRAINING FUNCTIONS =======================================

# Train a network on given training and validation data
def train(network,
          trn_data_frame,
          val_data_frame,
          predictand_std_data,
          lrn,
          epochs):

    trn_data_arr = trn_data_frame.to_numpy()
    trn_real_arr = dp.destd_predictands_range(trn_data_frame, predictand_std_data).to_numpy()
    trn_nrmse_arr = []

    val_data_arr = val_data_frame.to_numpy()
    val_real_arr = dp.destd_predictands_range(val_data_frame, predictand_std_data).to_numpy()
    val_nrmse_arr = []

    prev_nrmse_diff = 99999999999999

    for epoch in range (1, epochs+1):

        trn_predictions = destd_predict(network, trn_data_arr, predictand_std_data)
        trn_nrmse = rmse_calc(trn_predictions, trn_real_arr)
        trn_nrmse_arr.append(trn_nrmse)

        val_predictions = destd_predict(network, val_data_arr, predictand_std_data)
        val_nrmse = rmse_calc(val_predictions, val_real_arr)
        val_nrmse_arr.append(val_nrmse)

        current_nrmse_diff = abs(val_nrmse - trn_nrmse)
        if current_nrmse_diff > prev_nrmse_diff and epoch > 100:
            print(f"Terminated early at epoch {epoch}")
            return network, trn_nrmse_arr, val_nrmse_arr

        prev_nrmse_diff = current_nrmse_diff

        for row in trn_data_arr:
            network = backpropagate(network, row, lrn)


        if (epoch % 100) == 0:
            print(f"{epoch} epochs done")

    return network, trn_nrmse_arr, val_nrmse_arr



# Get predicted values from a network for each row in an array
def predict(network, data_arr):
    predicted_arr = []
    for row in data_arr:
        predicted_arr.append(forward_pass(network, row)[-1][0])
    return predicted_arr


# Predict dataset values and destandardise them
def destd_predict(network, data_arr, predictand_std_data):
    destd_predicted_arr = []
    for row in data_arr:
        output = forward_pass(network, row)[-1][0]
        destd_predicted_arr.append(dp.destd_val(output, predictand_std_data))
    return destd_predicted_arr


if __name__ == "__main__":
    test_data = np.array([[1, 0, 1]])

    test_network = [[[1, 3, 4], [-6, 6, 5]], [[-3.92, 2, 4]]]
    for epoch in range(1, 10001):
        for row in test_data:
            test_network, x = backpropagate(test_network, row, 0.1)
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}")


    print(f"Trained network prediction: {predict(test_network, test_data)}")