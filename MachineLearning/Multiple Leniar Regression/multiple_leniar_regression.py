import numpy as np
import pandas as pd
import os as o

o.chdir(r"E:\MachineLearning\DATASET\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 5 - Multiple Linear Regression")
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Encoding categorical data and must be done before splitting into 
# training and test set.
# here the categorical data variable col is "State"
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

# Avoiding Dummy Vairable Trap
# We only take n-1 dummay variables formed using the category variables.
# Here we have California/New York/Florida so we only take 2
# The current python will take of dummy variable trap but some software might
# so explictely do it

"""
In the below we are taking all rows and columns but starting from column index
=1 and leaving out col index =0

"""
x = x[:,1:]
# Splitting Dataset into Training Set and Test Set

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling

"""
# We dont need to take care of feature scaling in Multiple Leniar Regression as
  the library will handle it automatically.

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Building the Optimal model using Backward Elimination

import statsmodels.formula.api as sm

"""
# we need to add a column of 1's as statsmodel library will not treat MLR 
# as y = b0+b1x1+bx2 without the constant.
# we have to include the column of constant of 1
# will use the numpy to add a column in the beginning of the column
"""
# x = np.append(arr = x, values = np.ones((50,1)).astype(int), axis = 1)

"""
Definition : ones(shape, dtype=None, order='C')
ones(): This function will add value 1's
shape : number of rows and cols (not the value)
arr : this will hold array like values

Now the above functoin will append the col with value 1 in the end of the x dataset
To add we the col with 1 in the begining of the data set we need to do below
"""
# the below code will add dataset x to array of 50 cols with val 1
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)

# BE algo implementation
"""
Backward Elimination:(step by step method)
  1. select significance level
  2. fit the full model with all possible predictors
  3. Consider the predictor with highest P-value. If P>SL goto 4 else goto FIN
  4. Remove the predictor
  5. Fit the model without variable and we should rebuild the model again
     after rebuilding the model we have to goto 3 and repeat
FIN : Model ready
"""
x_opt = x[:,[0, 1, 2, 3, 4, 5]]
# SL = 0.05 or 5%
# we will remove the colmuns whose P-val > 0.05 or 5%

# Fit MLR model to future optimal model using statsmodel lib
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()
"""
OLS : Ordinarily squared model
Definition : OLS(endog, exog=None, missing='none', hasconst=None, **kwargs)

"""

# 2ND Iteration by removing the highest pval independant variable

x_opt = x[:,[0, 1, 3, 4, 5]]
# SL = 0.05 or 5%
# we will remove the colmuns whose P-val > 0.05 or 5%

# Fit MLR model to future optimal model using statsmodel lib
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

# 3 iteration by removing the highest pval independant val again

x_opt = x[:,[0, 3, 4, 5]]
# SL = 0.05 or 5%
# we will remove the colmuns whose P-val > 0.05 or 5%

# Fit MLR model to future optimal model using statsmodel lib
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

# 4th Iteration

x_opt = x[:,[0, 3, 5]]
# SL = 0.05 or 5%
# we will remove the colmuns whose P-val > 0.05 or 5%

# Fit MLR model to future optimal model using statsmodel lib
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

# 5th Iteration

x_opt = x[:,[0, 3]]
# SL = 0.05 or 5%
# we will remove the colmuns whose P-val > 0.05 or 5%

# Fit MLR model to future optimal model using statsmodel lib
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()