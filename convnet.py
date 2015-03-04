import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from processing import Data, HAND_DRAWN_DIR, RAND_ECOMPS_DIR

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

# kinda like l2 decay...
def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, w, w2, w3, w4, w_o, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    # problem happening here
    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

def testmodel(X, w, w2, w_o, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1b = max_pool_2d(l1a, (2, 2))
    l1 = T.flatten(l1b, outdim=2)
    l1 = dropout(l1, p_drop_conv)

    l2 = rectify(T.dot(l1, w4))
    l2 = dropout(l2, p_drop_hidden)
    py_x = softmax(T.dot(l2, w_o))

    # l2a = rectify(conv2d(l1, w2))
    # l2 = max_pool_2d(l2a, (2, 2))
    # l2 = dropout(l2, p_drop_conv)

    # l3a = rectify(conv2d(l2, w3))
    # l3b = max_pool_2d(l3a, (2, 2))
    # l3 = T.flatten(l3b, outdim=2)
    # l3 = dropout(l3, p_drop_conv)

    # problem happening here
    # l4 = rectify(T.dot(l3, w4))
    # l4 = dropout(l4, p_drop_hidden)

    # pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l2, l2, pyx    

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

trX, teX, trY, teY = trXteXtrYteY(
        use_hand_drawn=True,
        use_rand_ecomps=False,
        train_size_hand_drawn=0.7,
        train_size_rand_ecomps=0.3)

# trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 100, 100)
teX = teX.reshape(-1, 1, 100, 100)


# mnist images are 28 x 28
# trX = trX.reshape(-1, 1, 28, 28)
# teX = teX.reshape(-1, 1, 28, 28)

X = T.ftensor4()
Y = T.fmatrix()

# Construct the first convolutional pooling layer:
# filtering reduces the image size to (100-3+1 , 100-3+1) = (98, 98)
# maxpooling reduces this further to (98/2, 98/2) = (49, 49)
# 4D output tensor is thus of shape (batch_size, nkerns[0], 98, 98)
w = init_weights((32, 1, 3, 3))
w2 = init_weights((32 * 3 * 3, 25))
w_o = init_weights((25, 3))


# w = init_weights((32, 1, 3, 3))
# w2 = init_weights((64, 32, 3, 3))
# w3 = init_weights((128, 64, 3, 3))
# w4 = init_weights((128 * 3 * 3, 625))
# w_o = init_weights((625, 3))

"""
Current Error Statuses
ValueError: Shape mismatch: x has 18432 cols (and 128 rows) but y has 1152 rows (and 625 cols)
Apply node that caused the error: Dot22(Elemwise{mul,no_inplace}.0, <TensorType(float32, matrix)>)
Inputs shapes: [(128, 18432), (1152, 625)]
Inputs strides: [(73728, 4), (2500, 4)]
"""
noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = testmodel(X, w, w2, w_o, 0.2, 0.5)
l1, l2, l3, l4, py_x = testmodel(X, w, w2, w_o, 0., 0.)
y_x = T.argmax(py_x, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == predict(teX))
