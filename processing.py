import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage import exposure
from skimage.filter import threshold_otsu, gabor_filter
from sklearn.cross_validation import train_test_split

from utils import Utils

HAND_DRAWN_DIR = os.path.join(os.path.abspath('.'), 'smarterboard-images/')
RAND_ECOMPS_DIR = os.path.join(os.path.abspath('.'), 'rand-ecomps-images/')

class Preprocessing:
    @staticmethod
    def binary_from_thresh(img):
        """ Base function for converting to binary using a threshold """
        thresh = threshold_otsu(img)
        binary = img < thresh
        return binary

    @staticmethod
    def angle_pass_filter(img, frequency, theta, bandwidth):
        """ returns the magnitude of a gabor filter response for a certain
         angle """
        real, imag = gabor_filter(img, frequency, theta, bandwidth)
        mag = np.sqrt(np.square(real) + np.square(imag))
        return mag

    @staticmethod
    def frontslash_filter(img, denom, freq, bandwidth):
        """ intensifies edges that look like a frontslash '/' """
        theta = np.pi*(1.0/denom)
        return Preprocessing.angle_pass_filter(img, freq, theta, bandwidth)

    @staticmethod
    def backslash_filter(img, denom, freq, bandwidth):
        """ intensifies edges that look like a backslash '\' """
        theta = np.pi*(-1.0/denom)
        return Preprocessing.angle_pass_filter(img, freq, theta, bandwidth)


class FeatureExtraction:

    denom = 4.0
    freq = 0.50
    bw = 0.80

    @staticmethod
    def mean_exposure_hist(nbins, *images):
        """ calculates mean histogram of many exposure histograms
        args:
            nbins: number of bins
            *args: must be images (ndarrays) i.e r1, r2
        returns:
            histogram capturing pixel intensities """
        hists = []
        for img in images:
            hist, _ = exposure.histogram(img, nbins)
            hists.append(hist)
        return np.sum(hists, axis=0) / len(images)

    @staticmethod
    def mean_exposure_hist_from_gabor(img, nbins):
        frontslash = Preprocessing.frontslash_filter(
            img,
            FeatureExtraction.denom,
            FeatureExtraction.freq,
            FeatureExtraction.bw
        )
        backslash = Preprocessing.backslash_filter(
            img,
            FeatureExtraction.denom,
            FeatureExtraction.freq,
            FeatureExtraction.bw
        )
        return np.array(
            FeatureExtraction.mean_exposure_hist(nbins, frontslash, backslash)
        )

    @staticmethod
    def rawpix_nbins(image, nbins):
        """
        extracts raw pixel features and a histogram of nbins
        args:
            image: a m x n standardized shape ndarray representing an image
            nbins: nbins for histogram
        """
        gabor_hist = FeatureExtraction.mean_exposure_hist_from_gabor(
            image, nbins
        )
        image = image.flatten()
        return Utils.hStackMatrices(image, gabor_hist)


class Data:

    @staticmethod
    def getTrainFilenames(n, dir_path=HAND_DRAWN_DIR):
        filenames = os.listdir(dir_path)
        np.random.shuffle(filenames)
        filenames = filenames[:n]
        return filenames

    @staticmethod
    def isResistorFromFilename(filenames):
        is_resistor = [fn[0] == "r" for fn in filenames]
        return is_resistor

    @staticmethod
    def isComponentFromFilename(filenames):
        """ Parses directory of resistor and capacitor images and determines what
        the expected output should be.

        Arguments
        ---------
        filenames: a str 
            the path to the training image directory.  

        Returns
        -------
        Y: array-like, shape (n_samples, 3)
            labels or teaching examples
        """
        is_resistor = [fn[0] == "r" for fn in filenames]
        is_capacitor = [fn[0] == "c" for fn in filenames]
        is_inductor = [fn[0] == "i" for fn in filenames]
        Y = np.column_stack((is_resistor,is_capacitor,is_inductor))
        return Y

    @staticmethod
    def loadImageFeatures(filename, nbins):
        image = Data.loadImage(filename)
        return FeatureExtraction.rawpix_nbins(image, nbins)

    @staticmethod
    def loadImage(filename, square=True):
        if filename[-3:] == 'jpg':
            image = imread(filename, as_grey=True)
        elif filename[-3:] == 'npy':
            image = np.load(filename)
        if square:
            sqr_image = resize(image, (100, 100))
            return Preprocessing.binary_from_thresh(sqr_image)

        else:
            return Preprocessing.binary_from_thresh(image)

    @staticmethod
    def loadTrain(dir_path=HAND_DRAWN_DIR, oneHot=True):
        """ loads training data (trX, trY) for the nnet theano implementation. 
        See dinopants174/SmarterBoard for implementation including loading histograms of 
        Gabor Filtered Images 

        Arguments
        ---------
        dir_path: a str or None
            the path to the training image directory.  If None, uses the HAND_DRAWN_DIR path
            specified in processing.py.

        oneHot: boolean for resistor vs. resistor, capacitor, inductor testing

        Returns
        -------
        X: array-like, shape (n_samples, n_features)
            data inputs

        Y: array-like, shape (n_samples, 1)
            labels or teaching examples
        """

        fns = Data.getTrainFilenames(-1, dir_path)
        
        images = [np.ravel(Data.loadImage(dir_path + fn)) for fn in fns]
        X = np.vstack(images)
        
        if not oneHot:
            # y has shape (y.size,)
            y = np.array(Data.isResistorFromFilename(fns))
            # Y has shape (y.size, 1)
            Y = y.reshape(y.size, 1)
        else:
            Y = np.array(Data.isComponentFromFilename(fns))

        return X, Y

    @staticmethod
    def loadTrainTest(train_size, dir_path=HAND_DRAWN_DIR):
        """ loads training data, and holds out a percentage of this data for
        test validation """
        
        X, Y = Data.loadTrain(dir_path)

        trX, teX, trY, teY = train_test_split(X, Y, train_size=train_size)

        return trX, teX, trY, teY


def test_loadImage():
    import matplotlib.pyplot as plt
    resistor_path = HAND_DRAWN_DIR + 'resistor1.jpg'
    img = Data.loadImage(resistor_path, square=True)
    print img
    plt.imshow(img, cmap='gray')
    plt.title("Should be Square")
    plt.show()
    hist, bins = exposure.histogram(img)
    plt.plot(bins, hist)
    plt.show()

def test_loadTrainTest():
    import matplotlib.pyplot as plt
    trX, teX, trY, teY = Data.loadTrainTest(0.8, RAND_ECOMPS_DIR)
    img = teX[0].reshape((100, 100))
    print img
    img_label = "resistor" if teY[0, 0] == 1 else "capacitor"
    plt.imshow(img, cmap='gray')
    plt.title("Should be %s" % img_label)
    plt.show()
    hist, bins = exposure.histogram(img)
    plt.plot(bins, hist)
    plt.show()

def test_isComponentFromFilename():
    data = Data.getTrainFilenames(-1,HAND_DRAWN_DIR)
    result = Data.isComponentFromFilename(data)
    print result


if __name__ == '__main__':
    test_loadImage()
    # test_loadTrainTest()
    # test_isComponentFromFilename()
