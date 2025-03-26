import data_processing as dp
import training as trn
import pandas as pd
import numpy as np

# ============================== CONFIG ===============================================

# Network will have the same number of nodes per hidden layer and 1 output node
epochs = 1000
lrn_param = 0.1
hidden_layers = 1
nodes_per_layer = 5

# Name of the dataset to train the model on
# Must exist as a csv in /datasets/processed
dataset_name = "baseline"

# Name of the file to save the model in
model_name = "baseline_1000epochs"



# ============================= MODEL TRAINING ===========================================

# Load the dataset and standardisation metadata
dataset = dp.read_processed_csv(dataset_name)

training_data= dataset.loc[["trn"]]
validation_data = dataset.loc[["val"]]

# Initialise the neural network
network = trn.initialise_network(training_data.shape[1] - 1, hidden_layers, nodes_per_layer)
print("Network created")

# Train using backpropagation
print(f"Begin training...")
network, trn_output_arr, val_output_arr = trn.train(network, training_data, validation_data, lrn_param, epochs)
print("Training complete")

# Calculate root-mean-square error of training and validation data after every epoch
print("Calculating root-mean-square error at each epoch...")

predictand_std_data = dataset.loc["meta_std"].iloc[:, -1]
trn_rmse_arr = dp.rmse_calc(trn_output_arr, training_data, predictand_std_data)
val_rmse_arr = dp.rmse_calc(val_output_arr, validation_data, predictand_std_data)

print("Calculations done")

# Save the model and root-mean-square timelines
dp.write_model_json(network, trn_rmse_arr, val_rmse_arr, model_name)