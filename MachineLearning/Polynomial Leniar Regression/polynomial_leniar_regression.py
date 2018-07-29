# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 21:15:23 2018

@author: LENOVO
"""
import pandas as pd
import os as o
import matplotlib.pyplot as plt
import numpy as np

o.chdir(r"E:\MachineLearning\DATASET\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 6 - Polynomial Regression")

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

"""
Since we olny one col to be included into x and y
If we execute the above lines of code then x and y will be vectors.
But we will need our x to be matrix of features.
op : x size : (10,)
y : size (10,)
"""
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""
The above will change x to matrix and index 2 will be excluded
x is now matrix size (10,1)

"""

# Splitting Training Set and Test Set

"""
We should not split into Training and Test Set
 1. When we have small number of observations
 2. When we want to make very accurate predictions (also dataset is small)
     2a. We need more datasample as possible
"""

# Feature Scaling
"""
Not Required for PLR since we use same SLR Libraray
"""

# Build Leniar Regression Model and PLR

"""
Why create 2 models:
    TO compare both the models
"""
# Simple Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Polynomial Linear Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)

"""
Definition : PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)

Since there is only one independent variable we need to transform to poly eq
This will transform x with 1 independent variable into matrix of features x, x^2, x^3 ......
degree param = 3 then poly eq = bo + b1x1 + b2x1^2 + b3x1^3
since we are transforming the x we have to use .fittransform()
x_poly is the new x after transformation so we have again compare SLR an PLR
"""

# The lib has included the value 1 (constant in eq y = b0 + b1x1 ...)
# We will again create SLR on x_poly


# Computing and Viusalizing the SLR results:
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(x_poly, y)
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Salary Vs Experience Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Computing and Viusalizing the PLR results:
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_poly.predict(poly_reg.fit_transform(x)) , color='blue')
plt.title('Salary Vs Experience Polynomial Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Viusalize the PLR (temp) results:
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_poly.predict(x_poly) , color='blue')
plt.title('Salary Vs Experience Polynomial Linear Regression 2')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Computing and Viusalizing the PLR results:
# To improvve more accuracy :=change the degree in Poly regression

poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(x_poly, y)
# Viusalize the PLR (temp) results:
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_poly.predict(x_poly) , color='blue')
plt.title('Salary Vs Experience Polynomial Linear Regression 2')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""
The graph in above plt will is plotting straight line between points on x.
To have real cont curve we can icremented by higher resolution by 0.1

"""
x_grid = np.arange(min(x), max(x), 0.1)
# to convert to matrix of features
x_grid = x_grid.reshape(len(x_grid), 1)

# transform and predict using new x ie x_grid

x_grid_fit_transform_predict = lin_reg_poly.predict(poly_reg.fit_transform(x_grid))
plt.scatter(x, y, color='red')

plt.plot(x_grid, x_grid_fit_transform_predict , color='blue')
plt.title('Salary Vs Experience Polynomial Linear Regression 2')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting the salary with SLR
lin_reg.predict(6.5)


# Predicting the salary with SLR
lin_reg_poly.predict(poly_reg.fit_transform(6.5))


















