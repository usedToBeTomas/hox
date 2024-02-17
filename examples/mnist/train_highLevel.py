from hox import *
import hox.utils as utils

#Create model
model = Model.create([Dense(784, 144, Relu()), Dense(144, 10, Sigmoid())])

#Upload mnist dataset
X, Y, x, y = utils.mnist()

#Train the model
utils.train(model, X, Y, epochs = 2, rate = 1, batch_size = 32)
