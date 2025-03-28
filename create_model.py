import data_processing as dp
import training as trn
import numpy as np
import math

# ============================== CONFIG ===============================================

# Name of the file to save the model in
model_name = "baseline_2layers"

# Network will have the same number of nodes per hidden layer and 1 output node
epochs = 1000
lrn_param = 0.1
hidden_layers = 2

# Name of the dataset to train the model on
# Must exist as a csv in /datasets/processed
dataset_name = "baseline"




# ============================= MODEL TRAINING ===========================================

# Load the dataset and standardisation metadata
dataset, std_metadata = dp.read_processed_csv(dataset_name)
# Reverse applied functions to obtain the real predictand values
predictand_col = dp.destd_frame(dataset, std_metadata).iloc[:, -1]
raw_training_predictands = predictand_col.loc["trn"].values
raw_val_predictands = predictand_col.loc["val"].values

# Get split datasets and standardisation metadata
training_data= np.array(dataset.loc[["trn"]])
validation_data = np.array(dataset.loc[["val"]])
test_data = np.array(dataset.loc[["test"]])
input_size = training_data.shape[1] - 1

best_performance = 99999999999
best_nodes = 0
best_network = []
best_trn_predictions = []
best_val_predictions = []

#for nodes_per_layer in range(math.floor(input_size/2), input_size*2):
for nodes_per_layer in range(5,6):
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

    # Extract validation predictions for each row made by the trained network
    raw_trained_val_predictions = dp.destd_array(val_predictions, std_metadata.iloc[:, -1])

    # Calculate error functions on the final predictions
    error_vals = trn.epoch_error_calcs(raw_trained_val_predictions[-1], raw_val_predictands)

    # Compare to previous best RMSE
    performance = error_vals.loc["RMSE"].iloc[0]
    print(f"Validation RMSE: {performance}\n")
    if performance < best_performance:

        # Extract training predictions for each row made by the best network
        raw_trained_trn_predictions = dp.destd_array(trn_predictions, std_metadata.iloc[:, -1])

        best_performance = performance
        best_nodes = nodes_per_layer
        best_network = trained_network
        best_trn_predictions = raw_trained_trn_predictions
        best_val_predictions = raw_trained_val_predictions

print(f"Best network: {best_nodes}")
print(f"Best network RMSE: {best_performance}\n")

# Hyperparam metadata
hyperparams = {
    "dataset": dataset_name,
    "epochs": epochs,
    "lrn_param": lrn_param,
    "hidden_layers": hidden_layers,
    "nodes_per_layer": best_nodes,
    "val_rmse": best_performance
}

# Save the best model and root-mean-square timelines
dp.write_model_json(best_network, best_trn_predictions, best_val_predictions, hyperparams, model_name)