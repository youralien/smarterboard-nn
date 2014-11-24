import numpy as np
from foxhound.utils import floatX
from foxhound.neural_network.layers import Input, Dense
# from foxhound.utils import updates
from foxhound.neural_network.nnet import Net
from sklearn import metrics

from processing import Data

trX, teX, trY, teY = Data.loadTrainTest(0.9)

trX = floatX(trX)
teX = floatX(teX)
trY = floatX(trY)
teY = floatX(teY)

"""
###Notes on Layers:
* the shape of trY determines size of the layers.
* trY must be a matrix (not a vector).
* if outputs for each example is just a value
  (in the case of regression or bce, the matrix
  must be (n_examples, 1))
"""
layers = [
    Input(shape=trX[0].shape),
    Dense(size=512),
    Dense(size=512),
    Dense(activation='sigmoid')
]


# update = updates.Adadelta(regularizer=updates.Regularizer(l1=1.0))

model = Net(layers=layers, cost='bce', update='adadelta', n_epochs=5)
model.fit(trX, trY)
"""
###Note about predicts

* if doing binary classification ('bse', trY.shape = (trY.size, 1)),
	must do predict_proba and then np.round.
	model.predict uses np.argmax.
"""
print metrics.accuracy_score(teY, np.round(model.predict_proba(teX)))
