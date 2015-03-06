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

def init_weights(shape,std_dev=0.05):
    return theano.shared(floatX(np.random.randn(*shape) * std_dev))

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

def testmodel(X, w, w2, w3, w_o, p_drop_conv, p_drop_hidden, bias_1, bias_2, bias_3):
    l1a = rectify(conv2d(X, w, border_mode='valid')+bias_1)
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2)+bias_2)
    l2b = max_pool_2d(l2a, (2, 2))
    l2 = T.flatten(l2b, outdim=2)
    l2 = dropout(l2, p_drop_conv)

    l3 = rectify(T.dot(l2, w3)+bias_3)
    l3 = dropout(l3, p_drop_hidden)
    
    pyx = softmax(T.dot(l3, w_o))
    # l3a = rectify(conv2d(l2, w3))
    # l3b = max_pool_2d(l3a, (2, 2))
    # l3 = T.flatten(l3b, outdim=2)
    # l3 = dropout(l3, p_drop_conv)

    # problem happening here
    # l4 = rectify(T.dot(l3, w4))
    # l4 = dropout(l4, p_drop_hidden)

    # pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, pyx   

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
img_size  = 100
trX = trX.reshape(-1, 1, img_size,img_size)
teX = teX.reshape(-1, 1, img_size,img_size)

# mnist images are 28 x 28
# trX = trX.reshape(-1, 1, 28, 28)
# teX = teX.reshape(-1, 1, 28, 28)

X = T.ftensor4()
Y = T.fmatrix()

def get_reduced_img_size(img_size, kernel_size, border_mode='valid', downscale=2 ):
    """ Calculates the reduced image size after convolution
    """
    # Size adjustment after the convolution filter
    if border_mode=='valid':
        new_size = img_size - kernel_size + 1
    elif border_mode=='full':
	new_size = img_size + kernel_size + 1
    else:
	raise(ValueError, "border_mode must be 'valid' or 'full'")
    # Size adjustment after the maxpool step
    return np.ceil(float(new_size) / downscale)

kernel_size = 3
channels = 1
# Construct the first convolutional pooling layer:
# filtering reduces the image size to (100-3+1 , 100-3+1) = (98, 98)
# maxpooling reduces this further to (98/2, 98/2) = (49, 49)
# 4D output tensor is thus of shape (batch_size, nkerns[0], 49, 49)
n_fmaps = 32
bias_1 = init_weights((n_fmaps,98,98),std_dev = 0)

w = init_weights((n_fmaps, channels, kernel_size, kernel_size))
# img size determined by border_mode, see the first conv2d layer
# divided by 2 because of 2x2 maxpooling
reduced_img_size = get_reduced_img_size(img_size, kernel_size)

n_fmaps1 = 64
w2 = init_weights((n_fmaps1, n_fmaps, kernel_size, kernel_size))
bias_2 = init_weights((n_fmaps1,47,47),std_dev = 0)
reduced_img_size1 = get_reduced_img_size(reduced_img_size, kernel_size)
# the HiddenLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size, num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (batch_size, nkerns[1] * 49 * 49),
# or (128, 1 * 49 * 49) = (128, 2401) with the default values.
n_nodes_last_layer = 128
w3 = init_weights((n_fmaps1 * reduced_img_size1 * reduced_img_size1, n_nodes_last_layer))
bias_3 = init_weights((n_nodes_last_layer,),std_dev = 0)


n_out = 3
w_o = init_weights((n_nodes_last_layer, n_out))


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
noise_l1, noise_l2, noise_l3, noise_py_x = testmodel(X, w, w2, w3, w_o, 0.2, 0.5, bias_1,bias_2, bias_3)
l1, l2, l3, py_x = testmodel(X, w, w2, w3,  w_o, 0., 0., bias_1,bias_2, bias_3)
y_x = T.argmax(py_x, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w_o, bias_1, bias_2, bias_3]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(trY, axis=1) == predict(trX)), np.mean(np.argmax(teY, axis=1) == predict(teX))
