import pandas as pd
import numpy as np

from joblib import dump, load
from sklearn.preprocessing import StandardScaler

PATH = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
COLUMNS = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

class util:
    
    # Function: Feature Scaling
    @staticmethod
    def feature_normalize_train(X):
        scaler = StandardScaler()
        dump(scaler, 'model/scaler.joblib')
        return scaler.fit_transform(X)
    
    # Feature scaling during inference
    @staticmethod
    def feature_normalize_inference(X):
        scaler = load('model/scaler.joblib')
        return scaler.transform(X)
    
    @staticmethod
    def preprocess():
        autompg = pd.read_csv(PATH, sep='\s+', names = COLUMNS)
        autompg = autompg.iloc[:,:8]
        autompg = autompg.apply(pd.to_numeric, errors='coerce').dropna() # convert to numeric and drop rows with nan values
        # print(autompg.isna()) # check if data has invalid values

        X = autompg.iloc[:, 1:].values
        X = util.feature_normalize_train(X)
        x0 = np.ones(len(X))
        X = np.column_stack((x0, X))
        y = autompg[['mpg']].values
        y = y[:,0]
        return X, y