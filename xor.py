import training as trn
import numpy as np
import matplotlib.pyplot as plt

# Basic XOR model to demonstrate working backpropagation

xor_data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

xor_network = trn.initialise_network(2, 2, 4)
xor_network, trn_predictions, val_predictions = trn.train(xor_network, xor_data, xor_data, 0.1, 10000, trn.leaky_relu, trn.leaky_relu_derivative)

print(trn_predictions[-1])
print(xor_network)
trn_rmse_timeline = []
for epoch in range(1, 10001, 100):
    epoch_predictions = trn_predictions[epoch]
    epoch_rmse= trn.epoch_rmse_calc(epoch_predictions, [0,1,1,0])
    trn_rmse_timeline.append(epoch_rmse)
plt.plot(trn_rmse_timeline)
plt.xlabel('epochs (10^3)')
plt.ylabel('RMSE')
plt.show()