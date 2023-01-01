# Simple Linear Regression
rm(list = ls()) # Clears the environment

# Importing the dataset
dataset = read.csv('Salary_Data.csv') 


# Splitting the dataset into training and testing sets with caTools:

# install.packages('caTools') 
library(caTools)
set.seed(123) 

split = sample.split(dataset$Salary, SplitRatio = 2/3) # Split 
training_set = subset(dataset, split == TRUE) # Data got split and two thirds is stored here
test_set = subset(dataset, split == FALSE) # The remaining third was stored here


# Feature Scaling with Standardization - taken care of by the simple linear regression package