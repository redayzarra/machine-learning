# Data Pre-processing
rm(list = ls())


# Import the data set

dataset = read.csv('Data.csv')


# Addressing the missing data values for Age and Salary

dataset$Age = ifelse(is.na(dataset$Age), # Check if there are empty values in the Age column...
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)), # If there is a empty value in the Age column, then get the average with ave and then find the mean while also including the empty values.
                     dataset$Age) # If the value is not empty, then just leave it the way it is

dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)


# Encoding the Categorical Data (Country)

dataset$Country = factor(dataset$Country, # Factor function to replace the country names with specified codes
                         levels = c('France', 'Spain', 'Germany'), # Levels is the name of the categories we want to encode
                         labels = c(1, 2, 3)) # Labels is the number we are substituting them with

dataset$Purchased = factor(dataset$Purchased,
                           levels = c('Yes', 'No'),
                           labels = c(1, 0))


# Splitting the Data set into training and testing matrices with caTools

# install.packages('caTools')
library(caTools)

set.seed(123) # Specifies a seed so we can always replicate the random way the dataset was divided
split = sample.split(dataset$Purchased, SplitRatio = 0.8) # The sample.split method allows us to specify the column we will be applying the split to and then the ratio we will be using for the training set
training_set = subset(dataset, split == TRUE) # Set the variable training_set to be the subset of the split dataset
test_set = subset(dataset, split == FALSE) # The test_set needs to be false because we don't want the same split ratio


# Feature scaling the data with standardization

training_set[, 2:3] = scale(training_set[, 2:3]) # Scales the training_set matrix automatically however it only accepts numeric parameters. 
test_set[, 2:3] = scale(test_set[, 2:3]) # Since we used factors to encode our categorical data, we don't actually have ALL numeric values in our dataset so we will exclude the categorical columns