from hox import *
import utils

#Load model
model = Model.load("mnist")

#Upload mnist dataset
X, Y, x, y = utils.mnist()

#Accuracy tested on test10k data (x, y)
counter = 0
for i in range(len(x)):
    if np.argmax(model.forward(x[i])) == y[i]:
        counter +=1
print(str((counter*100)/len(y)) + "% accuracy")
