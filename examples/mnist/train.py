from hox import *
import utils

#Create model (2 layers, 784 input neurons, 144 first layer, 10 output layer)
model = Model.create([Dense(784, 144, Relu()), Dense(144, 10, Sigmoid())])

#Upload mnist dataset
X, Y, x, y = utils.mnist()

#Shuffle the dataset to improve training stability
indices = np.random.permutation(len(X))
X, Y = X[indices], Y[indices]

#Train the model
model.train(X, Y, epochs = 1, rate = 2, batch_size = 16)

#Save the trained model
model.save("mnist")
