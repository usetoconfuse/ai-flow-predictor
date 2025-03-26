import training as trn
import pandas as pd


# Basic XOR model to demonstrate working backpropagation

xor_data = pd.DataFrame([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

network = trn.initialise_network(2, 1, 4)
network, x, y = trn.train(network, xor_data, xor_data, 0.1, 10000)

print(trn.predict(network, xor_data))