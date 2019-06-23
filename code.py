"""Linear Regression with Gradient Descend
"""
__author__ = 'Bhavesh Kumar'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


'''
Function: Feature Scaling
'''
def feature_normalize(X):
    n_features = X.shape[1]
    means = np.array([np.mean(X[:,i]) for i in range(n_features)])
    stddevs = np.array([np.std(X[:,i]) for i in range(n_features)])
    normalized = (X - means) / stddevs
    return normalized

'''
Model Evaluation - Root Mean Square Error
Function: rmse
'''
def rmse(Y, Y_Pred):
    rmse = np.sqrt(sum((Y-Y_Pred)**2)/len(Y))
    return rmse

'''
Model Evaluate - R Square score
Function: r_sq_score
'''
def  r_sq_score(Y, Y_Pred):
    Y_Mean = np.mean(Y)
    sum_sq_Mean = sum((Y-Y_Mean)**2)
    sum_sq_Pred = sum((Y-Y_Pred)**2)
    r2 = 1 - (sum_sq_Pred/sum_sq_Mean)
    return r2

'''
Gradient Descent Algorithm function
Function: gradient_descent_algo
Input: X, Y, alpha - Learning Rate, iterations
Output: theta
'''
def gradient_descent_algo(X, Y, alpha, iterations):
    theta = np.zeros(X.shape[1])
    m = len(X)
    for i in range(iterations):
        gradient = (1/m) * np.matmul(X.T, np.matmul(X, theta) - Y)
        theta = theta - alpha * gradient
    return theta

'''
Data downloading and preprocessing
Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
Columns:
    1. mpg:           continuous
    2. cylinders:     multi-valued discrete
    3. displacement:  continuous
    4. horsepower:    continuous
    5. weight:        continuous
    6. acceleration:  continuous
    7. model year:    multi-valued discrete
    8. origin:        multi-valued discrete
    9. car name:      string (unique for each instance)

'''

column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
autompg = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', sep='\s+', names = column_names)
autompg = autompg.iloc[:,:8]
autompg = autompg.apply(pd.to_numeric, errors='coerce').dropna() # convert to numeric and drop rows with nan values
# print(autompg.isna()) # check if data has invalid values

X = autompg.iloc[:, 1:].values
X = feature_normalize(X)
x0 = np.ones(len(X))
X = np.column_stack((x0, X))
Y = autompg[['mpg']].values
Y = Y[:,0]

iterations = 1000
alpha = 0.01

theta = gradient_descent_algo(X, Y, alpha, iterations)

Y_Pred = X.dot(theta)
rmseValue = rmse(Y, Y_Pred)
r_sq_score = r_sq_score(Y, Y_Pred)

print('Using Custom library')
print('RMSE = %.10f \nR Square(R2) = %.10f'%(rmseValue,r_sq_score))
thetasym = chr(952)
print('Coefficients')
print(thetasym+'0 = %.15f'%(theta[0]))
print(thetasym+'1 = %.15f'%(theta[1]))
print(thetasym+'2 = %.15f\n-----'%(theta[2]))

'''
Validate with existing library
'''
reg = LinearRegression()
reg = reg.fit(X, Y)
slm_theta = reg.coef_
print(slm_theta)

# save model
dump(reg, 'model.joblib') 

# load model
reg = load('model.joblib') 
Y_pred = reg.predict(X)
rmseValue = np.sqrt(mean_squared_error(Y, Y_pred))
r_sq_score = reg.score(X, Y)

print('Using sklearn LinearRegression library')
print('RMSE = %.10f \nR Square(R2) = %.10f\n-----'%(rmseValue,r_sq_score))

# 3D Plot before

fig2 = plt.figure(figsize=(8,5))
ax2 = plt.axes(projection='3d')
displacement = X[:,1]
acceleration = X[:,2]
mpg = np.asarray(Y)
ax2.scatter(displacement, acceleration, mpg)
ax2.set_xlabel('Displacement')
ax2.set_ylabel('Acceleration')
ax2.set_zlabel('MPG')


# 3D Plot after

fig = plt.figure(figsize=(8,5))
ax = plt.axes(projection='3d')
displacement = X[:,1]
acceleration = X[:,2]
f1, f2 = np.meshgrid(displacement, acceleration)
Z = theta[0] + theta[1] * f1 + theta[2] * f2
ax.plot_surface(f1, f2, Z)
ax.set_xlabel('Displacement')
ax.set_ylabel('Acceleration')
ax.set_zlabel('MPG')
plt.show()
