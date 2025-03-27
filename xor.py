import training as trn
import numpy as np


# Basic XOR model to demonstrate working backpropagation

xor_data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

xor_network = trn.initialise_network(2, 1, 4)
xor_network, trn_predictions, val_predictions = trn.train(xor_network, xor_data, xor_data, 0.1, 10000)

print(trn_predictions[-1])