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

regressor = LinearRegression() # Assigning variable regressor as an instance object
regressor.fit(X_train, Y_train) # The fit() method will train the regression model and make calculations based on our training sets for the features (X_train) and our dependent variable (Y_train)


"""
Predicting the test set results by producing the observations from the test set.
The model should be able to accurately predict the salary of the test sets, that
we have set aside (there should be 6 of them), based on the years of experience.
The ground truth is the actual value of the salaries from our testing sets.
"""
Y_pred = regressor.predict(X_test) # Using the predict() method from the LinearRegression class, we can enter the testing set of our features as an argument to create an array of predictions. We will assign this new array to variable Y_pred 


"""
Visualizing the training set results with matplotlib with the pyplot module by 
creating a 2D plot with x-axis as the years of experience, and the y-axis being 
the salaries. We are creating this graph for the features training set or X_train.
"""
plt.scatter(X_train, Y_train, color = 'red') # Use the pyplot module, being shown as plt, and use the scatter() module to create a scatter plot. The x-axis is X_train and the y-axis is Y_train. We are also setting the color of the line to be red.
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # Use the plot method to plot the years of experience from our training set, and then uset the predict method to find the predicted salaries for the training set.

plt.title('Salary vs. Experience - Training Set') # Adding the title for the graph
plt.xlabel('Years of Experience') # Labeling the x-axis
plt.ylabel('Salary') # Labeling the y-axis

# plt.show() # Displaying the graph


"""
Visualizing the test set results with matplotlib, similar to what we did earlier.
We will use the same labels, and axis except now we will use the X_test and 
Y_test as our coordinates. However, the line will remain the same as the one
from the training set, so we won't change anything in plot.
"""
plt.scatter(X_test, Y_test, color = 'red') # The cooridnates of the test sets have to be used
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # We want the same line as our training sets so we will continue to use the same variables to make the same line.

plt.title('Salary vs. Experience - Test Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

# plt.show()


"""
Making a single prediction with the linear regression model that we have created.
We can accomplish this by using the predict() method as shown earlier and passing
a value for our x-axis or feature. In our case, the feature is years of 
experience and so we can input the value 15 (must be inside a 2D array like this
[[15]] because the predict() method only accepts 2D arrays).
"""
print(regressor.predict([[15]])) # Instead of giving an entire array to predict the values of, we are just giving one value on the x-axis and the predict() method will find the y-value on the line, or the salary (dependent variable).


"""
Finding the final linear regression equation is possible by directly getting the
values of the coefficients. We can get the values of the y-intercept (b0) and the
slope (b1) with the .intercept_ and the .coef_ method.
"""
coefficient = regressor.coef_ # Gets the coefficient (m) or the slope in the equation: y = mx + b 
intercept = regressor.intercept_ # Gets the y-intercept (b)

print(f'The equation of the line is: y = {coefficient}x + {intercept}') # Formatted string that displays the final regression line in one final equation!