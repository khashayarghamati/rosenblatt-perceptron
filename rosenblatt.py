import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


__author__ = 'Khashayar'
__email__ = 'khashayar@ghamati.com'


class Rosenblatt_Perceptron(object):

    def __init__(self, init_w, init_b):
        self.w = init_w
        self.b = init_b

    def estimate_w_and_b(self, train_data):
        """
        get a list and estimate y intercept and weights of classification line
        :param train_data:

        """

        # convert type of train data from list to pandas data frame
        sample_data = pd.DataFrame(train_data)

        # create a data frame from initial y intercept and weights
        w = pd.DataFrame([self.b] + self.w)

        # identify y sign
        y = w.T.dot(sample_data)
        y = self.sign(y.at[0, 0])

        # update w : w(n+1) = w(n) + .5[data_label - y] * train_data
        m = sample_data.at[2, 0] - y
        m *= .5
        c = sample_data.mul(m)
        new_w = w.add(c)
        self.w = new_w

    def sign(self, y):
        if y > 0:
            return 1
        else:
            return -1

    def get_w(self):
        return self.w

    def draw(self):
        x = np.array([self.w.at[1, 0], self.w.at[2, 0]])
        y = x + self.w.at[0, 0]
        plt.plot(x, y)
        plt.title("classification")
        plt.show()
