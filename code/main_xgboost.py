from src.constants import *
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Load data

data = pd.read_csv('data/preprocessed_train.csv')
evaluation = pd.read_csv('data/preprocessed_evaluation.csv')


# Check columns 
try:
    data.drop(columns = [date], axis = 1, inplace = True)
    evaluation.drop(columns = [date], axis = 1, inplace = True)
except : 
    print('nothing')

# Define variables 

train = data[[x for x in data.columns if x not in [ID]]]
predictors = [x for x in data.columns if x not in [ID,retweet_count]]
target = retweet_count


# Definition of pseudo-Huber loss
def huber_approx_obj(preds, ytrain):
    d = preds - ytrain 
    h = 1
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = - d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess

# Our optimal XGBoost

reg=xgb.XGBRegressor(objective=huber_approx_obj,
                         gamma = 7.4451354599911035,
                         n_estimators = 10,
                         max_depth = 14,
                         reg_lambda = 10,
                         min_child_weight=0)

# Fitting data

reg.fit(data[predictors],data[target])

# Predictions

pred = reg.predict(evaluation[predictors])
pred = np.around(pred)
res = pd.DataFrame({'TweetID':evaluation[ID], 'NoRetweets':pred})
res.to_csv('data/predictions.csv', index = False)
