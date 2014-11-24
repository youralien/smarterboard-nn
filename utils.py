import numpy as np


class Utils:

    @staticmethod
    def map_label_to_str(label):
        """
        args:
            label: boolean value returned by sklearns estimator.predict method
        returns:
            a string denoting the component name
        """
        if label is True:
            return 'resistor'
        else:
            return 'capacitor'

    @staticmethod
    def vStackMatrices(x, new_x):
        return Utils.stackMatrices(x, new_x, np.vstack)

    @staticmethod
    def hStackMatrices(x, new_x):
        return Utils.stackMatrices(x, new_x, np.hstack)

    @staticmethod
    def colStackMatrices(x, new_x):
        return Utils.stackMatrices(x, new_x, np.column_stack)

    @staticmethod
    def stackMatrices(x, new_x, fun):
        if x is None:
            x = new_x
        else:
            x = fun((x, new_x))

        return x
