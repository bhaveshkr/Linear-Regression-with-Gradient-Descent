import numpy as np
import pandas as pd

from joblib import dump, load
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from util import util

class model:
    def __init__(self):
        pass

    '''
    Model Evaluation - Root Mean Square Error
    Function: rmse
    '''
    def rmse(self, Y, Y_Pred):
        rmse = np.sqrt(sum((Y-Y_Pred)**2)/len(Y))
        return rmse

    '''
    Model Evaluate - R Square score
    Function: r_sq_score
    '''
    def  r_sq_score(self, Y, Y_Pred):
        Y_Mean = np.mean(Y)
        sum_sq_Mean = sum((Y-Y_Mean)**2)
        sum_sq_Pred = sum((Y-Y_Pred)**2)
        r2 = 1 - (sum_sq_Pred/sum_sq_Mean)
        return r2

    '''
    Gradient Descent Algorithm function
    Function: gradient_descent
    Input: X, Y, alpha - Learning Rate, iterations
    Output: theta
    '''
    def gradient_descent(self, X, Y, alpha, iterations):
        theta = np.zeros(X.shape[1])
        m = len(X)
        for i in range(iterations):
            gradient = (1/m) * np.matmul(X.T, np.matmul(X, theta) - Y)
            theta = theta - alpha * gradient
        return theta

    '''
    Train model
    '''
    def train(self, X, y):
        model = LinearRegression()
        fitted_model = model.fit(X, y)
        coefficients = fitted_model.coef_

        y_pred = fitted_model.predict(X)
        rmseValue = np.sqrt(mean_squared_error(y, y_pred))
        r_sq_score = fitted_model.score(X, y)

        # save model
        dump(fitted_model, 'model/model.joblib') 
        return 'RMSE = %.10f \nR Square(R2) = %.10f\nCoefficients = %s'%(rmseValue,r_sq_score, coefficients)

    '''
    Predict a value given input
    '''
    def predict(self, X):
        fitted_model = load('model/model.joblib')
        return 'Miles Per Gallon = %.10f'%(fitted_model.predict(X))