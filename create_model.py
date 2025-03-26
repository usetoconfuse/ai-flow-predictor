import data_processing as dp
import training as trn
import math

# ============================== CONFIG ===============================================

# Network will have the same number of nodes per hidden layer and 1 output node
epochs = 1000
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

training_data= dataset.loc[["trn"]]
validation_data = dataset.loc[["val"]]
predictand_std_data = dataset.loc["meta_std"].iloc[:, -1]
input_size = training_data.shape[1] - 1

best_performance = 99999999999
best_network = []
best_nodes = 0
best_trn_rmse_arr = []
best_val_rmse_arr = []

for nodes_per_layer in range(math.floor(input_size/2), input_size*2):

    # Initialise the neural network
    network = trn.initialise_network(training_data.shape[1] - 1, hidden_layers, nodes_per_layer)
    print(f"Network created with {nodes_per_layer} nodes per layer")

    # Train using backpropagation
    network, trn_rmse_arr, val_rmse_arr = trn.train(network,
                                                    training_data,
                                                    validation_data,
                                                    predictand_std_data,
                                                    lrn_param,
                                                    epochs)

    # Compare performance to previous best
    performance = val_rmse_arr[-1]
    print(f"Validation RMSE: {performance}\n")
    if performance < best_performance:
        best_performance = performance
        best_network = network
        best_nodes = nodes_per_layer
        best_trn_rmse_arr = trn_rmse_arr
        best_val_rmse_arr = val_rmse_arr

print(f"Best network: {best_nodes}\n")
print(f"Best network RMSE: {best_performance}\n")

# Save the best model and root-mean-square timelines
dp.write_model_json(best_network, best_trn_rmse_arr, best_val_rmse_arr, model_name)