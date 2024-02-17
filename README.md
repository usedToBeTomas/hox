# HOX

```pip install hox```


## examples/mnist
Mnist training, you have the option to utilize the high-level utils package for handling the training loop. Alternatively, you can choose to construct your own training loop using the low-level functions provided in hox. An example demonstrating low-level Mnist training can be found in the examples/mnist folder.
```python
from hox import *
import hox.utils as utils

#Create model
model = Model.create([Dense(784, 144, Relu()), Dense(144, 10, Sigmoid())])

#Upload mnist dataset
X, Y, x, y = utils.mnist()

#Train the model
utils.train(model, X, Y, epochs = 2, rate = 1, batch_size = 32)
```
