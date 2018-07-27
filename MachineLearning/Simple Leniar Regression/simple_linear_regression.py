# -*- coding: utf-8 -*-
"""
Created on Tue May  8 06:49:36 2018

@author: Yadukrishna
"""
"SIMPLE LINEAR REGRESSION"

# Importing the libraries

import numpy as np  # library contains mathematical tools
import matplotlib.pyplot as plt
import pandas as pd  # import/manage datasets
import os as o

# current directory is set
o.chdir(r"E:\MachineLearning\DATASET\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 4 - Simple Linear Regression\Simple_Linear_Regression")

# Part 2 - Regression\Section 4 - Simple Linear Regression)

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting Dataset into Training Set and Test Set

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Feature Scaling

"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""
from sklearn.linear_model import LinearRegression
regressorObj = LinearRegression()
regressorObj.fit(x_train, y_train)

# Predicting the TEST SET results

y_predicted = regressorObj.predict(x_test)

#   Visualizing the Training set results

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressorObj.predict(x_train), color='blue')
plt.title('Salary Vs Experience (Trainging Set)')
plt.xlabel('Yrs of Experience')
plt.ylabel('salary')
plt.show()

#   Visualizing the Test results

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressorObj.predict(x_train), color='blue')

"""
Why do we not have to change the x,y co-ordinates x_train,
regressorObj.predict(x_train) to x_test, regressorObj.predict(x_test)
Ans :
    Since we have already trained the LR object with the training set we will
    get similar (same) resluts even if we use x_test wih LR object
    So we dont need to change for the display of the test set results.
"""
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Yrs of Experience')
plt.ylabel('salary')
plt.show()
