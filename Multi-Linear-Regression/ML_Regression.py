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

  
"""