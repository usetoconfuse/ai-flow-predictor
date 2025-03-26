import training as trn
import numpy as np


# Basic XOR model to demonstrate working backpropagation

xor_data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

network = trn.initialise_network(2, 1, 4)
for epoch in range(1, 10001):
    for row in xor_data:
        network, x = trn.backpropagate(network, row, 0.1)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}")

print(trn.predict(network, xor_data))