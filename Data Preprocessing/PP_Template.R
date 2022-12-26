# Data Pre-processing Template
rm(list = ls())


# Importing the dataset
dataset = read.csv('Data.csv')
# dataset = dataset[, 2:3] Allows us to split the main dataset and only take what we need


# Splitting the dataset into training and testing sets with caTools

#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature Scaling with Standardization
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])