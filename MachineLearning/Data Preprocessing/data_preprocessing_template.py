# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 19:46:16 2018
1. library is tool to do specific job
@author: Administrator

IMPORTANT NOTES FOR Files:
    1. Need to disinguish between "MATRIX OF FEATURES" and "DEPENDENT VARIABLE VECTOR"
        Create Matrix of Features first
Matrix of Features :
    1. The list of columns that contains "INDEPENDENT VARIABLES" which should be
        processed including all lines (rows in the dataframe/table of dataset) in the dataset.
    2. The lines of Dataset (indexed rows of Table/dataframe) are called "Lines of Observation"

Dependent Variable Vector:
    1.
"""
# Data Preprocessing
# Importing the libraries

import numpy as np  # library contains mathematical tools
import matplotlib.pyplot as plt
import pandas as pd  # import/manage datasets
iport os as o

o.chdir(r"E:\MachineLearning\DATASET\Data_Preprocessing")
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values


"""
# We have created (Independent Features)Matrix of features (1st 3 columns)
# what is iloc
# what is [:,:-1] --> we take all the columns from dataset except the last one
[:,:-1]
1st part represents all rows
2nd part represents all columns expect the last one
"""
y = dataset.iloc[:, 3].values

"""
# we created Dependent variable vector
"""
testpractice = dataset.iloc[1, 0]

"""
The above line of code is for understating purpose
# testpractice variable is str variable which does not have .values attribute.
# [1,0] will return the value in 1st row and 0 column cell from"

"""
"""
Taking care of missing data:
    1. In the Matrix of Features we can always have missing data
    2. For Mathematical data missing in the columns we can use the "MEAN"
    techniqe to fill the missing DATA.
"""
# Import Imputer class from sklearn.preprocessing package
# create a object of Imputer class (ctrl + i) to help on Imputer class.
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

"""
Missing values will be represented by nan in the imported dataset
x[:,1:3] ==> check all the rows of columns 1 and 2; here upper bound 3 is
excluded.
        ==> This is python syntax
"""
"""
imputertest = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputertest = imputertest.fit(x[:, :])
x[:, :] = imputertest.transform(x[:, :])
"""

"""
We cannot use all the columns at once as they are string columns and others
are number columns.
 """

# Categorical Data

"""
In Machine Learning the model are based on Mathematics so we cant have text 
comuns in the dataset
    1. The text calumns generally are called Category Data
    2. We need to encode the categoery data so that we can 
    convert the text into numbers
"""

from sklearn import preprocessing

labelencoder_X = preprocessing.LabelEncoder()
x[:, 0] = labelencoder_X.fit_transform((x[:, 0]))

"""
What is the problem with above method of conversion of text to numbers.

    1. Since these are numbers there will be comparision between numbers and 
    since these are numbers which are encoded for countries.
    2. We have to make python equations to stopt making comparisions as there
    is relational comparision.
    3. For this we need use "DUMMY VARIABLES"
    4. We will create 3 more columns (depending on number of similar values in
    column)since we have France/Germany/Spain as common and every occurence of
    France in France column will be replaced with 1 and non occurence will be
    replaced with 0 , and same is repeated for other 2 columns as well.
    5. For this we will need "ONEHOTENCODER" class from sklearn.preprocessing
    pakage
"""
onehotencoder_obj = preprocessing.OneHotEncoder(categorical_features=[0])
x = onehotencoder_obj.fit_transform(x).toarray()

"""
Another categorical variables which we need to convert is the "Purchased"
Column.
We only need to perform "LabelEncoding" as this is a dependent variable
"""
labelencoder_Y = preprocessing.LabelEncoder()
y = labelencoder_Y.fit_transform(y)

# Splitting Dataset into Training Set and Test Set

"""
Need to split the data in training set and test set so that ML can understand
the corrleation between the independent variables and dependent variables in the
trainging set and can perform same on the test set

"""

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

"""
The above cross_validation class will be removed in favor of model_selection
and will need to be implemented like below

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
random_state=0)
"""

# Feature Scaling

"""
This is used reduce the Euclidian Distance (In General all ML algos are based
on EU Distance sqrt((x2-x1)^2 + (y2-y1)^2).

If in any of the features (columns or independent variables) have numerical
values whose difference is huge (observations or column values are not on the
same scale)then we will have to scale them so that we will not have
problem in ML models.
So we will have to reduce the EU distance between to points(here 2 columns).
2 Methods are used 
    1. Standard Reduction
    2. Normalised Reduction

In Standard Reduction : Xstand = (X - mean(X))/standardDeviation(X)
In Normalised Reduction : Xnorm = (X-min(X))/(max(X)-min(X))
    
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

"""
Why we did not fit transform test set of x?
since we have already fitted the training set to transform we dont need to do
it for the training set

Do we need to transform dummy variables (which we converted in categorical data
using labelEncodoer/onehotencoder)
This depends on the context as in this case if we scale the dummy variables we
will loose the interpretaion of which observation belongs to which country.
It will not break the Model if we dont scale the dummy variables.
For better accuracy we should scale them.

Why we did not feature scale y ?
No we dont featrue scale y as this dependent variable only takes 2 values 0 and
1
"""
