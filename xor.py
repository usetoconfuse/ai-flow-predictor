import training as trn
import data_processing as dp
import numpy as np

data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

network = trn.initialise_network(2, 1, 4)
for i in range(10000):
    network = trn.train(network, data, 0.1)

trn.predict(network, data)