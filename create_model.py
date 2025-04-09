import data_processing as dp
import training as trn
import numpy as np
import math

# ============================== CONFIG ===============================================

# Name of the file to save the model in
model_name = "baseline_batch_test"

# Name of the dataset to train the model on
# Must exist as a csv in /datasets/processed
dataset_name = "baseline"

activation_function = trn.leaky_relu
activation_function_derivative = trn.leaky_relu_derivative
momentum = False
weight_decay = False
bold_driver = False
annealing = False


# ============================= MODEL TRAINING ===========================================

# Load the dataset and standardisation metadata
dataset, std_metadata = dp.read_processed_csv(dataset_name)
pred_std_metadata = std_metadata.iloc[:, -1]
# Reverse applied functions to obtain the real predictand values
predictand_col = dp.destd_frame(dataset, std_metadata).iloc[:, -1]
raw_training_predictands = predictand_col.loc["trn"].values
raw_val_predictands = predictand_col.loc["val"].values

# Get split datasets and standardisation metadata
training_data= np.array(dataset.loc[["trn"]])
validation_data = np.array(dataset.loc[["val"]])
input_size = training_data.shape[1] - 1

best_performance = 99999999999
best_network = []
best_trn_predictions = []
best_val_predictions = []
hyperparams = {}

for epochs in [10000]:
    for lrn_param in [0.001, 0.01]:
        for hidden_layers in range(1, 2):
            for nodes_per_layer in range(math.floor(input_size/2), input_size*2):
                # Initialise the neural network
                network = trn.initialise_network(input_size, hidden_layers, nodes_per_layer)
                print(f"\nepochs: {epochs}"
                      f"\nlrn_param: {lrn_param}"
                      f"\nhidden_layers: {hidden_layers}"
                      f"\nnodes_per_layer: {nodes_per_layer}")

                # Train using backpropagation
                trained_network, trn_predictions, val_predictions = trn.train(network,
                                                                              training_data,
                                                                              validation_data,
                                                                              lrn_param,
                                                                              epochs,
                                                                              activation_function,
                                                                              activation_function_derivative,
                                                                              momentum,
                                                                              weight_decay,
                                                                              bold_driver,
                                                                              annealing)
                print("Training complete")



                # Compare performance to previous best

                # Extract validation predictions for each row made by the trained network
                raw_trained_val_predictions = dp.destd_array(val_predictions, pred_std_metadata)

                # Calculate error functions on the final predictions
                rmse = trn.epoch_rmse_calc(raw_trained_val_predictions[-1], raw_val_predictands)

                # Compare to previous best RMSE
                print(f"Validation RMSE: {rmse}")
                if rmse < best_performance:
                    print("New best!")

                    # Extract training predictions for each row made by the best network
                    raw_trained_trn_predictions = dp.destd_array(trn_predictions, pred_std_metadata)

                    best_performance = rmse
                    best_network = trained_network
                    best_trn_predictions = raw_trained_trn_predictions
                    best_val_predictions = raw_trained_val_predictions

                    # Hyperparameter metadata for best network
                    hyperparams = {
                        "dataset": dataset_name,
                        "epochs": epochs,
                        "lrn_param": lrn_param,
                        "hidden_layers": hidden_layers,
                        "nodes_per_layer": nodes_per_layer,
                        "val_rmse": best_performance,
                        "activation_function": activation_function.__name__,
                    }

print(f"\nBest network:"
      f"\n{hyperparams['epochs']} epochs"
      f"\n{hyperparams['lrn_param']} learning rate"
      f"\n{hyperparams['hidden_layers']} hidden layers"
      f"\n{hyperparams['nodes_per_layer']} nodes per layer"
      f"\n{hyperparams['val_rmse']} validation RMSE\n")

# Save the best model and root-mean-square timelines
dp.write_model_json(best_network, best_trn_predictions, best_val_predictions, hyperparams, model_name)