import hox
import gzip

#Define neural network model
model = Model.load("mnist")

#Upload mnist test dataset
X = np.frombuffer(gzip.open("t10k-images-idx3-ubyte.gz", "rb").read(), dtype=np.uint8)[16:].reshape((-1, 784)) / 255
Y = np.frombuffer(gzip.open("t10k-labels-idx1-ubyte.gz", "rb").read(), dtype=np.uint8)[8:]

#Accuracy test
counter = 0
for i in range(len(X)):
    if np.argmax(model.run(X[i])) == Y[i]:
        counter +=1
print(str((counter*100)/len(Y)) + "% accuracy")
