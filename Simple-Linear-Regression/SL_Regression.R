# Simple Linear Regression
rm(list = ls()) # Clears the environment

# Importing the dataset:
dataset = read.csv('Salary_Data.csv') 


# Splitting the dataset into training and testing sets with caTools:

# install.packages('caTools') 
library(caTools)
set.seed(123) 

split = sample.split(dataset$Salary, SplitRatio = 2/3) # Split 
training_set = subset(dataset, split == TRUE) # Data got split and two thirds is stored here
test_set = subset(dataset, split == FALSE) # The remaining third was stored here


# Feature Scaling with Standardization: taken care of by the simple linear regression package.


# Building the linear regression model using the built in simple linear regression package:

regressor = lm(formula = Salary ~ YearsExperience, # The lm function is used to fit linear models, it takes parameter formula which is the Salary is proportional to YearsExperience
               data = training_set) 

summary(regressor) # Code gives us valuable information on the regressor model. 
# We can see in the "Signif Codes" section that there is *** next to the independent variable, meaning the there is really high statistical significance.
# A p-value less than 5% means high statistical significance, meaning the more impact the independent variable will have on the dependent variable.


# Predicting the test results:

Y_pred = predict(regressor, newdata = test_set) # The predict function allows us to make predictions for any data, we have to pass in the model we built and the new data we want to predict on
Y_pred # The vector that stores our new predictions based off the test_set 


# Visualizing the training set results:

