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
        a. Select a significance level to stay in the model, maximum level.
        b. Fit the full model with all the possible predictors
        c. Consider the predictors with the highest p-value 
        d. Remove the highest predictors
        e. Fit the model without the variable (rebuild the whole model)
        f. Repeat until all variables p-values are lower than the maximum

  3. Forward Selection

  4. Bidirectional Elimination

  5. Score Comparison
"""