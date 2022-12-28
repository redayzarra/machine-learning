"""
Simple Linear Regression utilizes this linear equation: y = b0 + b1*x1 - where y
is the dependent variable (what we want to predict), and x1 is the independent
variable or feature, b0 is the y-intercept or a constant, and finally b1 is the
slope coefficient. Regression is when you want to predict a real value.

For example, predicting the output of potatoes from the amount of fertilizer a
farmer chooses to use. The amount of fertilizer is the independent variable or 
the feature (x1), while the dependent variable is the output of potatoes (y).
This would make the equation look like: Potatoes = b0 + b1*Fertilizer for our
linear regression model.

Ordinary Least Squares is a technique used to find the best regression line
through simple linear regression models. It works by finding the vertical 
distance from every data point(yi) to the regression line (ŷi). The difference
between these two (vertical distance between them) is called the residual: 
ε1 = yi - ŷi this can be used to find the best regression line. Using the linear
equation, we want to manipulate b0 and b1 in a way that the the SUM of the 
square of the residuals is as small as possible or SUM(yi - ŷi)^2 is smallest.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
Importing and reading the dataset with the help of pandas library
"""
dataset = pd.read_csv('Simple-Linear-Regression/Salary_Data.csv')
X = dataset.iloc[:, :-1].values # Assigning the variable X as the values of all the rows from all the columns except for the last one
Y = dataset.iloc[:, -1].values # Y variable is the dependent variable and it takes values of all rows from the last column

# print(X)
# print(Y)


"""
Splitting the dataset into the training set and the test set
"""
from sklearn.model_selection import train_test_split # Import the model selection module from sklearn library and use the train_test_split function to split the model into training and testing sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) # Establish the variables for the four different sets. Specifying the size of the test size to be 20% while the seed is at 0


# print(X_test)
# print(X_train)
# print(Y_test)
# print(Y_train)


"""
Building a simple linear regression model requires importing the correct class.
Simple linear regression models can be built by scratch or with libraries, in 
this case we will be using the scikit-learn library to build our model.
"""
from sklearn.linear_model import LinearRegression # Use the scikit-learn library to call the LinearRegression class from the linear model module

regressor = LinearRegression()
