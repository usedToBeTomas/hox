from hox import *
import hox.utils as utils

#Load model
model = Model.load("mnist")

#Upload mnist dataset
X, Y, x, y = utils.mnist()

#Accuracy test
counter = 0
for i in range(len(x)):
    if np.argmax(model.run(x[i])) == y[i]:
        counter +=1
print(str((counter*100)/len(y)) + "% accuracy")
