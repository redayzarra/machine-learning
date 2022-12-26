"""
Simple Linear Regression utilizes this linear equation: y = b0 + b1*x1 - where y
is the dependent variable (what we want to predict), and x1 is the independent
variable or feature, b0 is the y-intercept or a constant, and finally b1 is the
slope coefficient.

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