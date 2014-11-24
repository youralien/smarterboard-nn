import os
import cv2
import numpy as np
from skimage.transform import resize
from skimage import exposure
from skimage.filter import threshold_otsu, gabor_filter
from sklearn.cross_validation import train_test_split

from utils import Utils

TRAIN_DATA_DIR = os.path.join(os.path.abspath('.'), 'smarterboard-images/')
NUM_TRAIN = len(os.listdir(TRAIN_DATA_DIR))


class Preprocessing:
    @staticmethod
    def binary_from_thresh(img):
        """ Base function for converting to binary using a threshold """
        thresh = threshold_otsu(img)
        binary = img < thresh
        return binary

    @staticmethod
    def binary_from_laplacian(img):
        """ Function that converts an image into binary using
        a laplacian transform and feeding it into binary_from_thresh """
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        return Preprocessing.binary_from_thresh(laplacian)

    @staticmethod
    def scale_image(img, scaler, org_dim):
        """ resizes an image based on a certain scaler
        args:
            scaler: int or float. A value of 1.0 would output a
                image.shape = org_dim
            org_dim: tuple. Denoting (width, height)
        returns: ndarray. Scaled image """
        width, height = org_dim
        output_size = (int(scaler*width), int(scaler*height))
        return cv2.resize(img, output_size)

    @staticmethod
    def standardize_shape(img):
        """ standardizes the shape of the images to a tested shape for gabor
        filters """
        return Preprocessing.scale_image(img, scaler=.25, org_dim=(256, 153))

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
    def moments_hu(img):
        """
        returns the last log transformed Hu Moments
        args:
            img: M x N array
        """
        raw = cv2.HuMoments(cv2.moments(img))
        log_trans = -np.sign(raw)*np.log10(np.abs(raw))
        return log_trans.flatten()[-1]

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
    def getTrainFilenames(n):
        filenames = os.listdir(TRAIN_DATA_DIR)
        np.random.shuffle(filenames)
        filenames = filenames[:n]
        return filenames

    @staticmethod
    def isResistorFromFilename(filenames):
        is_resistor = [fn[0] == "r" for fn in filenames]
        return is_resistor

    @staticmethod
    def loadImageFeatures(filename, nbins):
        image = Data.loadImage(filename)
        return FeatureExtraction.rawpix_nbins(image, nbins)

    @staticmethod
    def loadImage(filename, square=True):
        image = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        if square:
            sqr_image = resize(image, (100, 100))
            return sqr_image
        else:
            return image

    @staticmethod
    def loadTrain(n, nbins):
        filenames = Data.getTrainFilenames(n)

        X = None

        for i in range(n):
            fn = filenames[i]
            X = Utils.vStackMatrices(
                X, Data.loadImageFeatures(TRAIN_DATA_DIR + fn, nbins)
            )

        y = Data.isResistorFromFilename(filenames)

        return np.array(X), np.array(y)

    @staticmethod
    def loadTrainTest(train_size):
        fns = Data.getTrainFilenames(-1)
        
        images = [np.ravel(Data.loadImage(TRAIN_DATA_DIR + fn)) for fn in fns]
        X = np.vstack(images)
        
        # y has shape (y.size,)
        y = np.array(Data.isResistorFromFilename(fns))
        # Y has shape (y.size, 1)
        Y = y.reshape(y.size, 1)

        trX, teX, trY, teY = train_test_split(X, Y, train_size=train_size)

        return trX, teX, trY, teY


def test_loadImage():
    import matplotlib.pyplot as plt
    # resistor_path = 'smarterboard-images/resistor1.jpg'
    resistor_path = '/home/rlouie/draw-rand-ecomps/img-sandbox/resistor10000.jpg'
    sqrimg = Data.loadImage(resistor_path, square=True)
    plt.imshow(sqrimg, cmap='gray')
    plt.title("Should be Square")
    plt.show()

def test_loadTrainTest():
    import matplotlib.pyplot as plt
    trX, teX, trY, teY = Data.loadTrainTest(0.8)
    img = teX[0].reshape((100, 100))
    img_label = "resistor" if teY[0] == 1 else "capacitor"
    plt.imshow(img, cmap='gray')
    plt.title("Should be %s" % img_label)
    plt.show()

if __name__ == '__main__':
    # test_loadImage()
    test_loadTrainTest()