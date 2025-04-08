import training as trn
import numpy as np
import matplotlib.pyplot as plt

# Basic XOR model to demonstrate working backpropagation
epochs = 10000

xor_data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

xor_network = trn.initialise_network(2, 3, 4)
xor_network, trn_predictions, val_predictions = trn.train(xor_network,
                                                          xor_data,
                                                          xor_data,
                                                          0.1,
                                                          epochs,
                                                          trn.leaky_relu,
                                                          trn.leaky_relu_derivative,
                                                          False,
                                                          False,
                                                          False,
                                                          False)

final_predictions = trn_predictions[-1]
print([float(x) for x in final_predictions])
trn_rmse_timeline = []
for epoch in range(epochs//100+1):
    epoch_predictions = trn_predictions[epoch]
    epoch_rmse= trn.epoch_rmse_calc(epoch_predictions, [0,1,1,0])
    trn_rmse_timeline.append(epoch_rmse)
plt.plot(trn_rmse_timeline)
plt.xlabel('epochs (10^2)')
plt.ylabel('RMSE')
plt.show()