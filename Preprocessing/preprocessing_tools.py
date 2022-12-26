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
X = dataset.iloc[: , :-1].values # Matrix of Features - Gets values from ALL ROWS, and all columns EXCEPT FOR LAST using Pandas' iloc function, which means locate indices, to single out the rows with our data and columns of features
Y = dataset.iloc[: , -1].values # Dependent Variable - Gets values from ALL ROWS, and ONLY LAST COLUMN using Pandas' iloc function

# print(X)
# print(Y)


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

imputer.fit(X[: , 1:3]) # The fit method from the SimpleImputer class allows to look for all the missing values in ALL ROWS and ONLY NUMERICAL COLUMNS
X[: , 1:3] = imputer.transform(X[: , 1:3]) # The transform function of the imputer object adds the mean values in the missing data, as specified by fit(), into a new dataset. We are rewritting the original dataset to include the transformed complete version

# print(X)


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

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough') # The column transformer, edits an entire column to our liking, takes two parameters... transformers - which is what we want to do, type of encoding we want to do, and on which columns. Remainders - the columns which we want to keep and don't want to apply transformations to.
X = np.array(ct.fit_transform(X)) # The Column Transformer class has a method called fit_transform which identifies the correct indices and changes it at the same time. However, the output is not an array which can be harmful for the model, so we use NumPy's array method to force the ouptut as an array. The output is a copy of X so we are reassigning it to the original X
# print(X)


"""
The dataset for the dependent variable also has to be encoded because the
dependent variable consists of Yes's and No's which can be rewritten as 0's and 
1's for the model to better understand.
"""
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() # Identifies Yes's and No's
Y = le.fit_transform(Y) # Automatically locates the indices of the Yes's and No's and transforms them into a new array with 1's and 0's

# print(Y)


"""
Splitting the dataset consists of making two seperate sets - one training set and
one test set which will be used for evaluation and depolyment. Splitting the data
is done before feature scaling, scaling all features to make them even, because
the test set is supposed to be a brand new set to which you should be evaluating
your machine learning model. You MUST treat it like a brand new dataset so you
cannot apply the same feature scaling to the testing set to prevent information
leakage.

We will use the train_test_split function from scikit-learn to divide the data
into four sections - features and dependent variable sets for both the training 
and testing sets. 
"""
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)


"""
Feature scaling allows us to scale our features evenly so some features do not
dominate other features (outliers). However, feature scaling is often unncessary
for most machine learning models because the models use linear regression 
equations, for example: y = b0 + b1x1 + b2x2 + ... , and so the model adjusts
for the features by choosing a smaller coefficient.

There are two main feature scaling techniques that put all features on the same
scale: standardization and normalization. Standardization is subtracting each 
value of the feature by the mean of the data and dividing the result by the 
standard deviation (results will be between -3 and 3). Normalization is when you 
subtract the minimum feature value from each feature value and then divide the 
result by the difference between the maximum and minimum feature value (results
will be between 0 and 1).

Normalization is recommended for a normal distribution in most of your features. 
Standardization is well-suited for most datasets and will work almost always 
which is why standardization is most-recommended. 
"""
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train[: , 3:] = sc.fit_transform(X_train[: , 3:])
