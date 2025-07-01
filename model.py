import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Lars, LarsCV, LassoLars, LassoLarsCV

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from walshbasis import WalshBasis

# Class to initialize model
class Model:
    def __init__(self, regression, n, k, interactions_nk, walsh_basis):
        # Regression type
        self.regression = regression

        # Number of variables
        self.n = n
        # Maximum interaction order
        self.k = k

        # Training data
        self.x_train = []
        self.y_train = []

        # Interactions as binary vector for given n and k
        self.interactions_nk = interactions_nk

        # Whether model uses Walsh basis (True or False)
        self.walsh_basis = walsh_basis

        # Makes polynomial features vector if not Walsh model
        if walsh_basis == False:
            self.poly = PolynomialFeatures(degree=self.k, interaction_only=True, include_bias=False)
            self.poly_fit = False

    # Add new training data and fit model
    def add_train_sample(self, x, y):
        x = np.atleast_2d(x)
        y = np.atleast_1d(y)

        if self.walsh_basis:
            new_x_train = []
            basis = WalshBasis(self.n, self.k, self.interactions_nk)
            for xi in x:
                xi_basis = basis.get_basis(xi)
                new_x_train.append(xi_basis)

        else:
            if not self.poly_fit:
                new_x_train = self.poly.fit_transform(x)
                self.poly_fit = True
            else:
                new_x_train = self.poly.transform(x)

        self.x_train.extend(new_x_train)
        self.y_train.extend(y)

        self.regression.fit(self.x_train, self.y_train)

    # Make a prediction
    def predicting(self, x):
        x = np.atleast_2d(x)

        x_test = []

        if self.walsh_basis:
            basis = WalshBasis(self.n, self.k, self.interactions_nk)
            for xi in x:
                xi_basis = basis.get_basis(xi)
                x_test.append(xi_basis)
        else:
            x_test = self.poly.transform(x)

        prediction = self.regression.predict(x_test)

        return prediction
    
