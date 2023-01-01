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

#install.packages("gglplot2") #Installs the ggplot2 library if it's not already there
library(ggplot2)

ggplot() + # The ggplot() section goes first and then we add the individual components afterwards like so:
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), # The geom_point function takes parameter aes which specifies what to plot on the X and Y axis
             color = 'red') + # Sets the color of the data points
  
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), # The geom_line function creates our regression line, however we will specify the y-axis to be the prediction values of the training set
            color = 'black') +
  
  ggtitle("Salary vs. Years of Experience - Training Set") +
  xlab("Years of Experience") +
  ylab("Salary")


# Visualizing the test set results:

ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             color = 'black') +
  
  geom_line(aes(x = test_set$YearsExperience, y = Y_pred),
            color = 'red') +
  
  ggtitle("Salary vs. Years of Experience - Test Set") +
  xlab("Years of Experience") +
  ylab("Salary")


# Finding the final coefficients:

coeffs <- regressor["coefficients"[1]]
coeffs <- coeffs$coefficients
b <- coeffs[1]
m <- coeffs[2]
