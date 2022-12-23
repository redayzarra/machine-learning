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


"""
The dataset does indeed have missing data and values for some of the features,
which can cause errors in our machine learning models. To address that, there
are certain measures we can take to fix that:
  1. Ignoring the missing data, simply removing it
  2. Replace the missing data with the average of all the data in the column [X]

Scikit-Learn is a data science library with a lot of tools, including data
preprocessing tools that aid us in replacing missing data.
"""
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') # Tells the imputer that the missing values we want to replace are the empty ones, and we want to replace them with the mean of the column

imputer.fit(x[: , 1:3]) # The fit method from the SimpleImputer class allows to look for all the missing values in ALL ROWS and ONLY NUMERICAL COLUMNS
x[: , 1:3] = imputer.transform(x[: , 1:3]) # The transform function of the imputer object adds the mean values in the missing data, as specified by fit(), into a new dataset. We are rewritting the original dataset to include the transformed complete version

print(x)


"""
The dataset includes categorical data that does not have a sequence of orders.
For example, in the first column we have the names of all the countries which
is important but has no numerical significance. We have to encode this data so
the model can interpret it correctly, and we do that with one hot encoding.

One hot coding means creating binary vectors for each different type of 
categorical data. For example, creating three different columns for 3 countries.
The purchased column which contrains Yes and No's will be converted into binary.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
