#!/usr/bin/env python3

class Model(object):
    def __init__(self):
        pass

    def train(self, X, y, l):
        """
        Objective:
        Minimize the loss function l of the model over data X with labels y.
         
        Example:

        Params:
        self: model to be trained
        X: kxn matrix (k datapoints each with n days of data)
        y: kx1 vector (k datapoints) in range {-1, 1}
            -1: price is lower/same+margin as final average price of X
            +1: price is higher (> same+margin) as final aver price of X
          
        """
        pass

    def predict(self, t, n, k, f):
        """
        Objective:
        Predict whether the time t+n (days) k-day average of a feature is
        higher or lower/same than the the time t k-day average.

        Example:
        t = 06-03-2011
        n = 365
        k = 5
        f = Google closing price
        Predict whether the average of Google's closing price from 
        06-01-2012->06-05-2012 is higher or lower/same than the average of 
        Google's closing price from 06-01-2011->06-05-2011.

        Params:
        self: model to be used for prediction
        t: starting datetime
        n: number of days to predict over (predict from t->t+n)
        k: number of days average is taken over (centered at datetime t)
        f: feature to predict over
        """
        pass
