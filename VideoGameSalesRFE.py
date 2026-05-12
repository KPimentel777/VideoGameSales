# Imports
import pandas                  as pd
import numpy                   as np
import statsmodels.api as sm
from   sklearn               import metrics
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.feature_selection import RFE

# Path and CSV
PATH = "C:/Users/soulr/Desktop/BCIT/COMP 4254 Advanced Topics/Assignment2/Dataset/"
CSV_DATA = "VideoGameSales.csv"
df = pd.read_csv(PATH + CSV_DATA)

# Show all columns and increase displayed amount
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# Impute the columns with missing or null values
# Makes sense for Year_of_Release and Critic_Score to be imputed with median of the respective column
# Makes sense for Critic_Count and User_Count to be imputed with 0 because it is valid
df['Year_of_Release'] = df['Year_of_Release'].fillna(df['Year_of_Release'].median())
df['Critic_Score'] = df['Critic_Score'].fillna(df['Critic_Score'].median())
df['Critic_Count'] = df['Critic_Count'].fillna(0)
df['User_Count'] = df['User_Count'].fillna(0)

# Convert 'User_Score' from a string to a float and fill in missing values with median
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
df['User_Score'] = df['User_Score'].fillna(df['User_Score'].median())

# Assign X and get dummy variables for Platform, Genre, Rating
X = df.copy()
X = pd.get_dummies(X, columns=['Platform', 'Genre', 'Rating'])

# Get dummy values on the top 10 Publishers, assigning all others to 'Other'
top_publishers = X['Publisher'].value_counts().nlargest(10).index
X['Publisher'] = X['Publisher'].where(X['Publisher'].isin(top_publishers),'Other')
X = pd.get_dummies(X, columns=['Publisher'], drop_first=True)

# Drop these columns as they are useless predictors
X = X.drop(columns=['Name', 'Developer'])
# Drop these columns (Global_Sales is our target and the other sales are redundant as they add up to Global_Sales)
X = X.drop(columns=['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])

# Identify float values
float_values = ['Year_of_Release', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']

# Convert all dummy columns to int from boolean
dummy_cols = X.columns.difference(float_values)
X[dummy_cols] = X[dummy_cols].astype(int)

# Assign our target variable
y = df['Global_Sales']

# Create the base model
model = LinearRegression()

# Choose amount of features HERE ******************
n_features = 1
rfe = RFE(estimator=model, n_features_to_select=n_features)
rfe.fit(X, y)

# Selected features printed here
selected_features = X.columns[rfe.support_]
print("\n*** SELECTED FEATURES (RFE) ***")
print(selected_features)

# Rank the features
ranking_df = pd.DataFrame({
    'Feature': X.columns,
    'Rank': rfe.ranking_
})
ranking_df = ranking_df.sort_values(by='Rank')
print("\n*** FEATURE RANKINGS ***")
print(ranking_df)

# Reassign the chosen features from RFE and add constant
X_selected = X[selected_features]
X_selected = sm.add_constant(X_selected)

# Fit the model
model = sm.OLS(y, X_selected).fit()

# Print summary
print(model.summary())

# RMSE
y_pred = model.predict(X_selected)

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y, y_pred))
print("\nRMSE:", rmse)