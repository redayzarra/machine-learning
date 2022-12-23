"""
Data Pre-Processing is the first stage of the machine learning process. Here, we
will:
  1. Import the data and all necessary libraries and modules
  2. Clean the data
  3. Split the data into training and test sets

Importing the necessary libraries and modules we need to start preprocessing 
our data. 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
Importing the dataset requires us to use the pandas library which will import
the dataset in a new variable. Then we have to create the matrix of features and
then the dependent variable vector. The features are the columns with which you
will predict the final decision (dependent variable). So, the dependent variable
is really what you WANT to predict.
"""
dataset = pd.read_csv('Preprocessing/Data.csv') # Pandas reads the dataset and creates a data frame. The data frame is stored in the variable dataset.
#   dataset.iloc[rows, columns]
x = dataset.iloc[: , :-1].values # Matrix of Features - Gets values from ALL ROWS, and all columns EXCEPT FOR LAST using Pandas' iloc function, which means locate indices, to single out the rows with our data and columns of features
y = dataset.iloc[: , -1].values # Dependent Variable - Gets values from ALL ROWS, and ONLY LAST COLUMN using Pandas' iloc function

print(x)
print(y)