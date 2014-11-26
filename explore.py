import numpy as np
from foxhound.utils import floatX
from foxhound.neural_network.layers import Input, Dense
# from foxhound.utils import updates
from foxhound.neural_network.nnet import Net
from sklearn import metrics
import matplotlib.pyplot as plt

from processing import Data, HAND_DRAWN_DIR, RAND_ECOMPS_DIR

def trXteXtrYteY(use_hand_drawn, use_rand_ecomps, train_size_hand_drawn, train_size_rand_ecomps):
    """
    Arguments
    ---------
    use_hand_drawn: boolean
        whether to include HAND_DRAWN_DIR

    use_rand_ecomps: boolean
        whether to include RAND_ECOMPS_DIR

    train_size_hand_drawn: float btwn 0-1
        percentage of hand drawn data for training

    train_size_rand_ecomps: float btwn 0-1
        percentage of rand ecomps data for training
    """

    if not use_hand_drawn and not use_rand_ecomps:
        print "Can't Get Data from Nowhere. Must use one dataset source"
        return None

    elif use_hand_drawn and not use_rand_ecomps:
        trX, teX, trY, teY = Data.loadTrainTest(train_size_hand_drawn, HAND_DRAWN_DIR)
        return trX, teX, trY, teY

    elif use_rand_ecomps and not use_hand_drawn:
        trX, teX, trY, teY = Data.loadTrainTest(train_size_rand_ecomps, RAND_ECOMPS_DIR)
        return trX, teX, trY, teY

    else:
        trX, teX, trY, teY = Data.loadTrainTest(train_size_hand_drawn, HAND_DRAWN_DIR)

        # Combine Random Computer Generated EComponents
        trX1, teX1, trY1, teY1 = Data.loadTrainTest(train_size_rand_ecomps, RAND_ECOMPS_DIR)
        trX = np.vstack((trX, trX1))
        teX = np.vstack((teX, teX1))
        trY = np.vstack((trY, trY1))
        teY = np.vstack((teY, teY1))

        return trX, teX, trY, teY

def is_normalized(trX):
    img = trX[0, :]
    print img
    if np.max(img) > 1:
        print "Not Normalized. Must Fall between 0 -1 "
        return False
    else:
        print "Check: is_normalized"
        return True

trX, teX, trY, teY = trXteXtrYteY(
    use_hand_drawn=True,
    use_rand_ecomps=False,
    train_size_hand_drawn=0.7,
    train_size_rand_ecomps=0.99)

print trY.shape

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
    Dense(size=2000, p_drop=0.5),
    Dense(size=400, p_drop=0.5),
    Dense(activation='sigmoid')
]


# update = updates.Adadelta(regularizer=updates.Regularizer(l1=1.0))

model = Net(layers=layers, cost='bce', update='adadelta', n_epochs=10)
model.fit(trX, trY)
"""
###Note about predicts

* if doing binary classification ('bse', trY.shape = (trY.size, 1)),
    must do predict_proba and then np.round.
    model.predict uses np.argmax.
"""

print "Train Accuracy"
print metrics.accuracy_score(trY, np.round(model.predict_proba(trX)))

print "Test Accuracy"
pred_proba = model.predict_proba(teX)

for example_idx in range(10):
    img = teX[example_idx, :].reshape((100, 100))
    plt.imshow(img, cmap='gray')
    plt.title("Resistor? {}".format(pred_proba[example_idx]))
    plt.show()

print metrics.accuracy_score(teY, np.round(pred_proba))