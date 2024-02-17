from hox import *
import hox.utils as utils

#Create model
model = Model.create([Dense(784, 144, Relu()), Dense(144, 10, Sigmoid())])

#Upload mnist dataset
X, Y, x, y = utils.mnist()

#Train the model
rate = 1
batch_size = 32
epochs = 2
optim = SGD(model, rate)
for epoch in range(epochs):
    print("epoch = " + str(epoch))
    for batch_start in range(0, len(X), batch_size):
        batch_end =  min(batch_start + batch_size, len(X))
        optim.zero_grad()
        for i in range(batch_start, batch_end):
            model.run(X[i])
            optim.grad(Y[i])
        optim.step()

#Save the model
model.save("mnist")
print("done")
