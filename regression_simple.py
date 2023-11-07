import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

from statsmodels.stats.diagnostic import linear_rainbow
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split
from statsmodels.stats.diagnostic import het_goldfeldquandt

url = "https://raw.githubusercontent.com/Alyssasorensen/datasci_7_regressions/main/datasets/U.S._Chronic_Disease_Indicators__CDI_%20(2).csv"
df = pd.read_csv(url)
df

### Dependent Variable: DataValue
## This variable represents the value of a particular metric or measurement. It can be considered the dependent variable because it's the value that you might want to predict or analyze in relation to other factors.

### Independent Variable: YearStart
## YearStart is a numerical variable that represents the starting year of a data record. It can be considered the independent variable in this context because it can be used to analyze how the dependent variable (DataValue) changes over different years.

# Filter the dataset to keep only the relevant columns
df = df[['YearStart', 'DataValue']]

# Drop rows with missing values
df.dropna(inplace=True)

X = df[['YearStart']]
y = df['DataValue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data Points')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression')
plt.xlabel('YearStart')
plt.ylabel('DataValue')
plt.legend()
plt.show()

# Check linearity (scatterplot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['YearStart'], y=y_test)
plt.title('Linearity Check')
plt.xlabel('YearStart')
plt.ylabel('DataValue')
plt.show()

# Check independence of errors (residual plot)
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.residplot(x=X_test['YearStart'], y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title('Residual Plot for Independence of Errors')
plt.xlabel('YearStart')
plt.ylabel('Residuals')
plt.show()

# Check constant variance (homoscedasticity)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.title('Homoscedasticity Check')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Check normality of errors (Q-Q plot)
import scipy.stats as stats
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Normality of Residuals (Q-Q Plot)')
plt.show()

