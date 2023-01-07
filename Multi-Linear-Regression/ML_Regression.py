"""
Multiple Linear Regression utilizes this equation for it's models and its general
theory: ŷ = b0 + b1*x1 + b2*x2 + ... + bn*xn - where ŷ represents the dependent
variable (what we want to predict) and the b0 represents the y-intercept. Next, the
bn represent the slope coffiecient and the xn is every dependent variable that will
be used in the dataset. The dependent variables (xn) are the features of the data 
and the equation will only have as many bn*xn components where n represents the 
number of features. 

For example, if we are growing potatoes on a farm we would need to account for the
amount of water we are using, the fertilizer amount, the time in the sun, etc. All
of these would be individual features that would affect the overall growth of our
potatoes and we would set each one as a bn*xn component in our equation. So our 
equation would be: Potatoes (ŷ) = 8 tons (b0)  + 3 tons Fertilizer + 2 tons H2O etc.


THE FIVE ASSUMPTIONS FOR LINEAR REGRESSION:

However, you cannot just blindly apply linear regression. You have to make sure 
that your dataset is fit for using a linear regression model. There are five main
assumptions to make before using linear regression:

  1. Linearity - making sure that there is a directly proportional relationship
  between each independent and dependent variable. There must be some form of 
  linear relationship for you to apply linear regression.

  2. Homoscedasticity - defined as equal variance. Meaning you don't want to see
  any sort of cone-like shape forming with your dataset because that would mean 
  that variance is dependent of the independent variable (where linear regression
  wouldn't be a good fit).

  3. Multivariate Normality - defined as normality of error distribution. Meaning,
  if your dataset has many trends (dataset forms multiple seperate lines or 
  trends) then linear regression is not a good fit. You want your line to be 
  intersecting as many points as possible or atleast in the right ballpark.

  4. Independence - being independent of observations including no autocorrelation.
  We don't want to see patterns in our data, the last thing we want to see is our
  datasets following some sort of wave-like patterns because this means that our
  independent variables (features) are not independent of each other. They are 
  somehow affecting one another which means linear regression is not an option.

  5. Lack of Multicollinearity - meaning we don't want our independent variables
  or predictors to not be correlated with each other. If they are not correlated 
  then we can build a linear regression model for the data and it would work. But, 
  if the data's independent variables are correlated then we can't use linear models.

While this is not one of the five assumptions, an extra check we can run when using
linear regression is an outlier check. An outlier check would find outliers in a 
dataset and remove them from significantly affecting the linear regression model.


DUMMY VARIABLES AND THEIR TRAP:

Dummy variables are used when we encounter categorical data in our dataset. For
example, if we are looking at the profit a company generates from spending in 
marketing, research, customer service.. we can just use the numerical values for
our x variable in our equation: ŷ = b0 + b1*x1 + b2*x2 ... + bn*xn - however, it is
difficult to add categorical variables such as state or region the company is 
located at. For this scenario, we use dummy variables. 

To use dummy variables, identify all the categories you have in your categorical
independent variable. If there are only companies in New York and Cali then you 
would have two categories. Take these categories and create new columns for them,
containing 1's and 0's for whether or not the data is from New York (1 if yes) or
not (0 if no). The equation would then include:  ŷ = b0 + bn*xn + bn*Dn - where Dn
is the dummy variables from your categorical data. 

However, it is not necessary to use all of your categories as dummy variables. For 
example, you can use just one dummy variable for New York (with 1's and 0's) 
because if it's not in NY then it has to be in CA. This is because we don't want
the linear regression model to include the coefficient for CA in the y-intercept.
When you are building a model, it is crucial to always remove one dummy variable 
because it causes multicollinearity since it creates two variables that affect 
each other in some way. If you have 9 dummy variables, you should only include 8. 


BUILDING A MULTI-LINEAR REGRESSION MODEL:

You may encounter data with many independent variables, and one of the first steps
for building a linear regression model is to decide which independent variables to
keep and which to throw out. It is necessary to discard some of your independent 
variables (features) because the model might suffer from having these extra 
variables to account for. 

There are five main methods to building multiple linear regression models:

  1. All In - these are cases where you would just blindly throw in all of your
  variables. You would do this if you have prior knowledge that these are variables
  you should be using. You would also be using this for preparing for backward
  elimination.

  2. Backward Elimination:
        a. Select a significance level to stay in the model, minimum level.
        b. Fit the full model with all the possible predictors
        c. Consider the predictors with the highest p-value 
        d. Remove the highest predictors
        e. Fit the model without the variable (rebuild the whole model)
        f. Repeat until all variables p-values are lower than the minimum level

  3. Forward Selection:
        a. Select the maximum significance level for predictors to enter the model
        b. Fit all the simple regression models (y ~ xn)
        c. Select the one with the lowest p-value
        d. Keep the variable and add on ONE other predictor
        e. Consider the predictor model with the lowest p-value
        f. Repeat adding variables until the p-value reaches the maximum level
        g. Keep the previous model not the current one

  4. Bidirectional Elimination: also known as Step-Wise Regression
        a. Select a significance level to enter and stay in the model
        b. Perform the next step of forward selection (new variables P < SL-ENTER)
        c. Perform ALL the steps of backward elimination (old variables P < SL-STAY)
        d. Repeat step c every time you grow the model by a variable
        e. No new variables can be added or no old variables can exit

  5. Score Comparison: All possible models (very resource intense)
        a. Select a criterion of goodness of fit (e.g Akaike criterion)
        b. Construct all possible regression models: 2^(n-1) total combinations
        c. Select the model with the best criterion


Importing the libraries we need for implementing the multi-linear regression models
with pandas, sci-kit learn, numpy, and matplotlib
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
We then import the dataset and read it with the pandas library. And then using the
.iloc method provided by the pandas library to assign the independent and dependent
variables on the dataset. The dependent variable is usually the last column, in our
case it is the "Profit" column.
"""
dataset = pd.read_csv('Multi-Linear-Regression/50_Startups.csv')

X = dataset.iloc[: , :-1].values
Y = dataset.iloc[: , 1].values

# print(X)
# print(Y)


"""
Our dataset contains categorical data as seen in the "State" column where there
are many categories (state names) where companies are located. To address this, we
will have to create dummy variables which will encode the categories and allow the 
model to interpret the data. 

We are able to use sci-kit learn to apply the one-hot encoding technique. We will 
need the ColumnTransformer class to change the columns in our dataset and then the 
OneHotEncoder to encode the categorical data. We need numpy to make sure the output
of the .fit_transform method from the ColumnTransformer class is an array.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# print(X)


"""
Splitting the dataset into the training and test set. We can use the sci-kit learn
model selection library to utilize the train_test_split function.
"""
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)


"""
Feature Scaling - There is no need to use feature scaling in multiple linear 
regression because the coefficients will balance themselves out to adjust for 
higher values.
"""


"""
The dummy variable trap and step-wise regression is not necessary to implement
multiple linear regression because the sci-kit learn class that we will use already
avoids the dummy variable trap (where categorical data needs to be encoded and one
category needs to be left out to avoid redundancies). The class will also perform 
step-wise regression for us and will automatically chooset the features with the
highest p-value to predict the dependent variable.
"""
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)


"""
Since there are many features on the dataset, it isn't possible to graph the model
results so instead we will create two vectors. One will contain a sample of the 
actual profit values, or the dependent variable, while the other vector contains a
sample of the predicted values from the model. We will compare the two vectors to
effectively evaluate our model.
"""
Y_pred = regressor.predict(X_test) # The .predict() method from the LinearRegression class predicts the profits (dependent variable) from the training features set (X_test) and stores it in Y_pred
np.set_printoptions(precision = 2) # Displays any numerical value with only 2 decimals

print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1)) # The concatenate function expects the tuple of arrays that we want to concatenate. Since we want to concatenate the Y_pred (or the predicted values for the depenedent variable), we will add Y_pred in the concatenate function. However, because we want to print the array vertically we can use the .reshape() method to print the vector vertically, which also takes the length of the array (because we need the number of elements in Y_pred) and the number of columns which is just 1. We apply the same .reshape method to the Y_test vector because we want to concatenate that. The concatenate function also needs an axis parameter which will be 1 because we want to concatenate horizontally.


"""
How to implement code to avoid the dummy variable trap (simply removing one 
categorical variable to avoid redundancy):
"""
# X = X[:, 1:] 


