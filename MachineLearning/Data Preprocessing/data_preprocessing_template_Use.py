# -*- coding: utf-8 -*-

# Data Preprocessing
# Importing the libraries

import numpy as np  # library contains mathematical tools
import matplotlib.pyplot as plt
import pandas as pd  # import/manage datasets
import os as o
# reading the dataset
o.chdir(r"E:\MachineLearning\DATASET\Data_Preprocessing")
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Splitting Dataset into Training Set and Test Set

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling

"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""
