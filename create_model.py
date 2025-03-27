import data_processing as dp
import training as trn
import numpy as np
import pandas as pd

# ============================== CONFIG ===============================================

# Network will have the same number of nodes per hidden layer and 1 output node
epochs = 100
lrn_param = 0.1
hidden_layers = 1

# Name of the dataset to train the model on
# Must exist as a csv in /datasets/processed
dataset_name = "firstimproved"

# Name of the file to save the model in
model_name = "firstimproved_basic"



# ============================= MODEL TRAINING ===========================================

# Load the dataset and standardisation metadata
dataset = dp.read_processed_csv(dataset_name)

# Reverse applied functions to obtain the real predictand values
predictand_col = dp.destd_frame(dataset).iloc[:, -1]
raw_predictands = predictand_col.values
raw_predictands = np.exp(raw_predictands)

# Get split datasets and standardisation metadata
training_data= np.array(dataset.loc[["trn"]])
validation_data = np.array(dataset.loc[["val"]])
test_data = np.array(dataset.loc[["test"]])
std_metadata = dataset.loc["meta_std"]
input_size = training_data.shape[1] - 1

best_performance = 99999999999
best_network = []
best_nodes = 0

#for nodes_per_layer in range(math.floor(input_size/2), input_size*2):
for nodes_per_layer in range(5, 6):
    # Initialise the neural network
    network = trn.initialise_network(training_data.shape[1] - 1, hidden_layers, nodes_per_layer)
    print(f"Network created with {nodes_per_layer} nodes per layer")

    # Train using backpropagation
    trained_network, trn_predictions, val_predictions = trn.train(network,
                                                                  training_data,
                                                                  validation_data,
                                                                  lrn_param,
                                                                  epochs)


    # Compare performance to previous best

    # Predictions for each row made by the trained network
    trained_val_predictions = dp.destd_array(val_predictions, std_metadata.iloc[:, -1])
    destd_val_predictions = trained_val_predictions[-1]
    raw_trained_val_predictions = np.exp(destd_val_predictions)

    # Calculate error functions on the final predictions
    error_vals = trn.epoch_error_calcs(raw_trained_val_predictions, raw_predictands)

    # Compare to previous best RMSE
    performance = error_vals.loc["RMSE"].iloc[0]
    print(f"Validation RMSE: {performance}\n")
    if performance < best_performance:
        best_performance = performance
        best_network = network
        best_nodes = nodes_per_layer

print(f"Best network: {best_nodes}\n")
print(f"Best network RMSE: {best_performance}\n")

# Save the best model and root-mean-square timelines
#dp.write_model_json(best_network, best_trn_rmse_arr, best_val_rmse_arr, model_name)