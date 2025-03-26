import data_processing as dp
import training as trn
import math

# ============================== CONFIG ===============================================

# Network will have the same number of nodes per hidden layer and 1 output node
epochs = 200
lrn_param = 0.1
hidden_layers = 1

# Name of the dataset to train the model on
# Must exist as a csv in /datasets/processed
dataset_name = "first_improved"

# Name of the file to save the best model in
model_name = "first_improved_1000epochs"



# ============================= MODEL TRAINING ===========================================

# Load the dataset and standardisation metadata
dataset = dp.read_processed_csv(dataset_name)

training_data= dataset.loc[["trn"]]
validation_data = dataset.loc[["val"]]
test_data = dataset.loc[["test"]]
predictand_std_data = dataset.loc["meta_std"].iloc[:, -1]

# Train neural networks with varying number of hidden nodes
best_network = []
best_performance = 99999999
best_node_num = 0
best_trn_nrmse_arr = []
best_val_nrmse_arr = []

input_size = training_data.shape[1]-1
for nodes_per_layer in range(math.floor(input_size/2), 2*input_size):
    # Initialise the neural network
    network = trn.initialise_network(training_data.shape[1] - 1, hidden_layers, nodes_per_layer)
    print(f"Hidden layers: {hidden_layers}, nodes per layer: {nodes_per_layer}")

    # Train using backpropagation
    network, trn_nrmse_arr, val_nrmse_arr = trn.train(network,
                                                      training_data,
                                                      validation_data,
                                                      predictand_std_data,
                                                      lrn_param,
                                                      epochs)

    performance = val_nrmse_arr[-1]
    print(f"Validation performance: {performance}\n")

    if performance < best_performance:
        best_performance = performance
        best_network = network
        best_node_num = nodes_per_layer
        best_trn_nrmse_arr = trn_nrmse_arr
        best_val_nrmse_arr = val_nrmse_arr

print(f"Best performing: {best_node_num}")

# Test best model against test data
test_data_arr = test_data.to_numpy()
test_real_arr = test_data_arr[:, -1]
test_predictions = trn.predict(best_network, test_data_arr)
test_nrmse = trn.rmse_calc(test_predictions, test_real_arr)

print(f"Best model's performance on test data: {test_nrmse}")

# Save the model and root-mean-square timelines
dp.write_model_json(best_network, best_trn_nrmse_arr, best_val_nrmse_arr, model_name)