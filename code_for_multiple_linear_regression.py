# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:15:04 2021

@author: asdg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
d1=[1,2,3,4];
d2=[2,4,6,8];
d3=[3,6,9,12];
d4=[4,8,16,32];
temp=[5,25,12,125];

x_train = [d1, d2,d3, d4]
y_train = temp;


#Splitting the dataset
from sklearn.model_selection import train_test_split 
#x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100);
#Fitting the Multiple Linear Regression model
mlr = LinearRegression()  
k=mlr.fit(x_train, y_train);

reg = LinearRegression().fit(x_train, y_train)
reg.score(x_train, y_train)
reg.coef_

#Intercept and Coefficient
print("Intercept: ", mlr.intercept_)
print("Coefficients:")
#Prediction of test set
y_pred_mlr= mlr.predict(x_test)
#Predicted values
print("Prediction for test set: {}".format(y_pred_mlr));

#----------------------multiple support vector linear regression------





'''
#Model Evaluation
from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('R squared: {:.2f}'.format(mlr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)'''