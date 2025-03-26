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
dataset_name = "test"

# Name of the file to save the model in
model_name = "test"



# ============================= MODEL TRAINING ===========================================

# Load the dataset and standardisation metadata
dataset = dp.read_processed_csv(dataset_name)

training_data = dataset.loc[["trn", "val", "test"]]

# Initialise the neural network
network = trn.initialise_network(training_data.shape[1] - 1, hidden_layers, nodes_per_layer)
print("Network created")

# Train using backpropagation
print(f"Begin training...")
network, output_arr = trn.train(network, training_data, lrn_param, epochs)
print("Training complete")



# Calculate root-mean-square error after every epoch
print("Calculating root-mean-square error at each epoch...")
output_df = pd.DataFrame(output_arr).transpose()
predictand_std_data = dataset.loc["meta_std"].iloc[:, -1]
output_df = dp.destd_predictands_range(output_df, predictand_std_data)

predictand_col = training_data.loc[:, (slice(None), "p")]
predictand_col = dp.destd_predictands_range(predictand_col, predictand_std_data)
real_vals = predictand_col.to_numpy()

rmse_arr = []
for epoch in range(epochs):
    epoch_predictions = output_df.iloc[:, epoch]
    total_sq_err = 0
    for row in range(len(epoch_predictions)):
        error = real_vals[row][0] - epoch_predictions[row]
        total_sq_err += error**2
    mse = total_sq_err / len(epoch_predictions)
    rmse_arr.append(np.sqrt(mse))
print("Calculations done")

dp.write_model_json(network, rmse_arr, model_name)