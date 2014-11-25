import numpy as np
from foxhound.utils import floatX
from foxhound.neural_network.layers import Input, Dense
# from foxhound.utils import updates
from foxhound.neural_network.nnet import Net
from sklearn import metrics
import matplotlib.pyplot as plt

from processing import Data, HAND_DRAWN_DIR, RAND_ECOMPS_DIR

trX, teX, trY, teY = Data.loadTrainTest(0.7, HAND_DRAWN_DIR)

# Combine Random Computer Generated EComponents
trX1, teX1, trY1, teY1 = Data.loadTrainTest(0.975, RAND_ECOMPS_DIR)
trX = np.vstack((trX, trX1))
teX = np.vstack((teX, teX1))
trY = np.vstack((trY, trY1))
teY = np.vstack((teY, teY1))

def is_normalized(trX):
    img = trX[0, :]
    print img
    if np.max(img) > 1:
        print "Not Normalized. Must Fall between 0 -1 "
        return False
    else:
        print "Check: is_normalized"
        return True

assert is_normalized(trX)

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
    Dense(size=512, p_drop=0.2),
    Dense(size=512, p_drop=0.4),
    Dense(activation='sigmoid', p_drop=0.5)
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
pred_proba = model.predict_proba(teX)
print pred_proba[:10]

for example_idx in range(10):
    img = teX[example_idx, :].reshape((100, 100))
    plt.imshow(img, cmap='gray')
    plt.title("Resistor? {}".format(pred_proba[example_idx]))
    plt.show()

print metrics.accuracy_score(teY, np.round(pred_proba))
