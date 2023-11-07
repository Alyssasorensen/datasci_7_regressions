import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

url = "https://raw.githubusercontent.com/Alyssasorensen/datasci_7_regressions/main/datasets/Cancer_Rates%20(1).csv"
df = pd.read_csv(url)
df

## Dependent Variable (Y):

## Dependent Variable: All_Cancer

## Independent Variables (X):

# Independent Variable 1: Colorectal
# Independent Variable 2: Lung_Bronc
# Independent Variable 3: Breast_Can
# Independent Variable 4: Prostate_C
# Independent Variable 5: Urinary_Sy

# Define the dependent variable (Y) and independent variables (X)
Y = df['All_Cancer']
X = df[['Colorectal', 'Lung_Bronc', 'Breast_Can', 'Prostate_C', 'Urinary_Sy']]

# Add a constant term (intercept) to the model
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(Y, X).fit()

# Get the model summary
print(model.summary())

# Calculate VIF for each independent variable
vif_data = pd.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Print the VIF data
print("\nVIF Data:")
print(vif_data)

## Results Interpreted:

# Constant (Intercept) VIF = 17.071047

# The constant term, or intercept, typically has a high VIF because it is not related to any other variable in the model. You can ignore its VIF value.

# Colorectal, VIF = 3.098965:

# The VIF of Colorectal is relatively low (below 5), suggesting that it has little multicollinearity with the other variables in the model.

# Lung_Bronc, VIF = 5.131104:

# The VIF of Lung_Bronc is moderate (around 5), indicating some degree of multicollinearity with other variables but not extremely high.

# Breast_Can, VIF = 2.754088:

# The VIF of Breast_Can is low (below 5), indicating little multicollinearity with other variables.

# Prostate_C, VIF = 3.697852:

# The VIF of Prostate_C is relatively low (below 5), suggesting limited multicollinearity.

# Urinary_Sy, VIF = 4.272061:

# The VIF of Urinary_Sy is moderate (around 4), indicating some multicollinearity but not extremely high.

# Define the independent variables (e.g., X1, X2, X3)
X1 = df['Colorectal']
X2 = df['Lung_Bronc']
X3 = df['Breast_Can']

# Define the dependent variable (Y)
Y = df['All_Cancer']

# Create scatterplots
plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.scatter(X1, Y)
plt.xlabel('Colorectal')
plt.ylabel('All_Cancer')

plt.subplot(132)
plt.scatter(X2, Y)
plt.xlabel('Lung_Bronc')
plt.ylabel('All_Cancer')

plt.subplot(133)
plt.scatter(X3, Y)
plt.xlabel('Breast_Can')
plt.ylabel('All_Cancer')

plt.tight_layout()
plt.show()

# Fit the multiple linear regression model
model = sm.OLS(Y, X).fit()

# Get the fitted values (predicted values) from the model
fitted_values = model.fittedvalues

# Create a scatterplot of observed vs. fitted values
plt.figure(figsize=(10, 6))
plt.scatter(Y, fitted_values, c='blue', label='Observed vs. Fitted Values')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2, color='red', label='Ideal Line')
plt.xlabel('Observed Values (Y)')
plt.ylabel('Fitted Values')
plt.title('Observed vs. Fitted Values')
plt.legend()
plt.grid(True)
plt.show()

## Assessing normality of the residuals
W, p_value = shapiro(residuals)
print(f"Shapiro-Wilk Test: W={W}, p-value={p_value}")

# Plot Q-Q plot of residuals
plt.figure(figsize=(10, 6))
stats.probplot(residuals, plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()

## Assessing the homogeneity of variance of the residuals

# Fit the multiple linear regression model
model = sm.OLS(Y, X).fit()

# Get the residuals from the model
residuals = model.resid

# Create a Residuals vs. Fitted Values plot
plt.figure(figsize=(10, 6))
sns.residplot(x=model.fittedvalues, y=residuals, lowess=True, scatter_kws={'alpha': 0.5})
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.grid(True)
plt.show()

# Plot residuals vs fitted values

# Fit the multiple linear regression model
model = sm.OLS(Y, X).fit()

# Get the residuals from the model
residuals = model.resid

# Create a Residuals vs. Fitted Values plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=model.fittedvalues, y=residuals, color='blue', alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.grid(True)
plt.show()

