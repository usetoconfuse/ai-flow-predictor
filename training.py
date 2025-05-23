import numpy as np
import pandas as pd

# =========================== INITIALISE A NEURAL NETWORK =======================================

# Generate weights of the neural network using Xavier normal initialization
# size: number of nodes on this layer
# inputs: number of nodes on previous layer
def weighted_layer(size, inputs):
    lim = np.sqrt(6/(inputs+size))
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
    for i in range(hidden_layers - 1):
        network.append(weighted_layer(layer_size, layer_size))
    # Output layer (1 node)
    network.append(weighted_layer(1, layer_size))

    return network



# ====================== ERROR CALCULATION =======================================


# Calculate RMSE from an array of predicted values and an array of real values
def epoch_rmse_calc(predicted_vals, real_vals):
    total_sq_err = 0
    for row in range(len(predicted_vals)):
        total_sq_err += (predicted_vals[row] - real_vals[row])**2

    mse = total_sq_err / len(predicted_vals)
    return np.sqrt(mse)


# Calculate different error functions on an array of predicted values against an array of real values
# Returns a DataFrame of error function values
def epoch_error_calcs(predicted_vals, real_vals):

    total_sq_err = 0
    total_sq_rel_err = 0
    total_real_sq_mean_diff = 0
    total_pred_sq_mean_diff = 0
    total_diff_product = 0

    mean_real_value = np.mean(real_vals)
    mean_pred_value = np.mean(predicted_vals)

    for row in range(len(predicted_vals)):

        obs = real_vals[row]
        pred = predicted_vals[row]

        err = pred - obs
        total_sq_err += err**2
        total_sq_rel_err += (err / obs)**2
        total_real_sq_mean_diff += (obs - mean_real_value)**2
        total_pred_sq_mean_diff += (pred - mean_pred_value)**2
        total_diff_product += (obs - mean_real_value) * (pred - mean_pred_value)

    mse = total_sq_err / len(predicted_vals)
    rmse = np.sqrt(mse)

    msre = total_sq_rel_err / len(predicted_vals)

    ce = 1 - (total_sq_err / total_real_sq_mean_diff)

    r_sq_denom = np.sqrt(total_real_sq_mean_diff * total_pred_sq_mean_diff)
    r_sq = (total_diff_product / r_sq_denom)**2

    error_frame = pd.DataFrame(data=[rmse, msre, ce, r_sq],
                        index=["RMSE", "MSRE", "CE", "R^2"],
                        columns=["Error Values"])

    return error_frame



# =============================== ACTIVATION FUNCTIONS =======================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return max(x, 0)

def relu_derivative(x):
    if x > 0:
        return 1
    else:
        return 0

def leaky_relu(x):
    if x > 0:
        return max(x, 0)
    else:
        return 0.01 * x

def leaky_relu_derivative(x):
    if x > 0:
        return 1
    else:
        return 0.01

#######################################################################################################
# =============================== TRAIN NETWORK AND BACKPROP ==========================================
#######################################################################################################

def forward_pass(network, row, activation):
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
            u = activation(node_val)

            outputs.append(u)
        inputs = outputs
        u_vals.append(outputs)

    return u_vals



# Get predicted values from a network over np array
def predict(network, data_arr, activation):
    pred_arr = []
    for row in data_arr:
        pred_arr.append(forward_pass(network, row, activation)[-1][0])
    return pred_arr



# Backpropagation: forward pass and backward pass over a row of data
def backpropagate(network,
                  row,
                  lrn_param,
                  activation,
                  derivative,
                  net_changes=None,
                  beta=None,
                  omega=None):



    predictand = row[-1]

    # FORWARD PASS
    u_vals = forward_pass(network, row, activation)


    # BACKWARD PASS

    # Calculate delta at the output node
    output = u_vals[-1][0]
    error = predictand - output

    # Weight decay: penalise large weights on output node
    if beta and omega:
        error += beta * omega

    delta_output = error * derivative(output)

    deltas = [[delta_output]]

    # Iterate through hidden layers in reverse to get deltas
    for layer in range(len(u_vals) - 2, -1, -1):
        layer_deltas = []
        total_layer_delta = 0

        # For each node in this layer, calculate the delta as the mean of deltas to each node in the next layer
        for current_node_pos in range(len(network[layer])):
            total_node_delta = 0
            u = u_vals[layer][current_node_pos]

            # Delta from this node to each node in the next layer
            for output_node_pos in range(len(network[layer+1])):
                weight_to_output = network[layer+1][output_node_pos][current_node_pos+1]
                total_node_delta += weight_to_output * delta_output * derivative(u)

            mean_node_delta = total_node_delta
            layer_deltas.append(mean_node_delta)
            total_layer_delta += mean_node_delta

        # Prepend to deltas list as we are iterating backwards
        deltas.insert(0, layer_deltas)

        # The output delta for the next layer is the sum of the mean deltas of all nodes in this layer
        delta_output = total_layer_delta

    # Prepend input layer values for weight adjustment calculations
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
            bias_change = lrn_param * node_delta
            bias_momentum = 0.9 * net_changes[l][n][0]
            new_node = [node[0] + bias_change + bias_momentum]

            # For each input weight on this node from a node in the previous layer,
            # get the output value u_val of the node in the previous layer
            # and use this to build the new set of weights for this node
            for weight in range(1, len(node)):

                u = u_vals[l][weight-1]
                weight_change = lrn_param * node_delta * u
                weight_momentum = 0.9 * net_changes[l][n][weight]
                new_node = new_node + [(node[weight] + weight_change + weight_momentum)]

            new_layer.append(new_node)
        new_network.append(new_layer)

    # Return the updated network
    return new_network



# Train a network on given training and validation data as np arrays
# Store the rmse for each epoch
def train(network,
          trn_data,
          val_data,
          lrn_param,
          epochs,
          activation,
          derivative,
          momentum,
          weight_decay,
          bold_driver,
          annealing):

    # Store predicted values of all rows before training
    trn_prediction_arr = [predict(network, trn_data, activation)]
    val_prediction_arr = [predict(network, val_data, activation)]

    # Network changes from last epoch - all 0s to begin with
    net_changes = []
    for layer in range(len(network)):
        net_changes.append(np.zeros_like(network[layer]))

    # Initial omega value for weight decay
    omega = None
    n_weights = None
    if weight_decay:
        n_weights = 0
        omega = 0

        for layer in range(len(network)):
            for node in range(len(network[layer])):
                n_weights += len(network[layer][node])



    # Training RMSE last epoch - for bold driver
    if bold_driver:
        prev_trn_rmse = 999999999999999


    epoch = 1
    while epoch <= epochs:

        retries = 0

        # Annealing: automatically adjust wait based on epoch from 0.1 to 0.01
        if annealing:
            lrn_param = 0.01 + ((0.1 - 0.01) * (1 - (1 / (1 + (pow(np.e, 10 - ((20 * epoch) / epochs)))))))

        beta = None
        if weight_decay:

            # Regularisation parameter
            beta = 1/(epoch * lrn_param)

            # Omega parameter
            for layer in range(len(network)):
                for node in range(len(network[layer])):
                    omega += np.sum(np.square(network[layer][node]))
            omega = omega / (2*n_weights)


        # Iterate over all rows in the dataset for this epoch
        rows = len(trn_data)
        row = 0
        while row < rows:

            new_network = backpropagate(network,
                                        trn_data[row],
                                        lrn_param,
                                        activation,
                                        derivative,
                                        net_changes,
                                        beta,
                                        omega)



            # Calculate the changes in weights this epoch
            if momentum:
                for layer in range(len(network)):
                    net_changes[layer] = np.subtract(new_network[layer], network[layer])



            # Bold driver every 200 epochs
            if bold_driver and epoch % 200 == 0:

                # Calculate training data predictions after this update
                row_trn_predictions = predict(new_network, trn_data, activation)

                # Bold driver: if RMSE of training data has increased by over 4%,
                # undo weight changes and decrease lrn
                row_trn_rmse = epoch_rmse_calc(row_trn_predictions, trn_data[:, -1])
                if row_trn_rmse > prev_trn_rmse * 1.04 and lrn_param > 0.01:
                    lrn_param = max(lrn_param * 0.7, 0.01)
                    retries += 1
                else:
                    row += 1
                    lrn_param = min(lrn_param * 1.05, 0.5)

                    # Save updated network
                    network = new_network
                    prev_trn_rmse = row_trn_rmse
            else:
                network = new_network
                row += 1

        # Store the predicted values of all rows after every 100 epochs

        if (epoch % 100) == 0:
            epoch_trn_predictions = predict(network, trn_data, activation)
            epoch_val_predictions = predict(network, val_data, activation)

            trn_prediction_arr.append(epoch_trn_predictions)
            val_prediction_arr.append(epoch_val_predictions)
            print(f"{epoch} epochs done")

        if bold_driver and epoch % 200 == 0:
            print(f"Retried rows {retries} times on epoch {epoch}")

        epoch += 1

    return network, trn_prediction_arr, val_prediction_arr

#######################################################################################################
# =============================== END OF TRAIN NETWORK AND BACKPROP ===================================
#######################################################################################################