# Data Pre-processing

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