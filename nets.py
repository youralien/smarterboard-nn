import numpy as np
from foxhound.utils import floatX
from foxhound.neural_network.layers import Input, Dense
from foxhound.utils import updates
from foxhound.neural_network.nnet import Net
from sklearn import metrics
from processing import Data


trX, teX, trY, teY = Data.loadTrainTest(0.9)

trX = floatX(trX)
teX = floatX(teX)
trY = floatX(trY)
teY = floatX(teY)


layers = [
        Input(shape=trX[0].shape,
        Dense(size = 512),
        Dense(size = 512).
        Dense(activation = 'sigmoid')
        ]


model = Net(layers=layers, cost='bce', update='adadelta', n_epochs=5)
model.fit(trX, trY)

