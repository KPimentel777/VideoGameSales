# Imports
import pandas                  as pd
import numpy                   as np
from   sklearn               import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# Path and CSV
PATH = "C:/Users/soulr/Desktop/BCIT/COMP 4254 Advanced Topics/Assignment2/Dataset/"
CSV_DATA = "VideoGameSales.csv"
df = pd.read_csv(PATH + CSV_DATA)

# Show all columns and increase displayed amount
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

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

# Reassign X to statistically significant variables (found using RFE, FF, FI)
X = X[['Publisher_Nintendo', 'Critic_Count', 'User_Count', 'Critic_Score', 'Year_of_Release', 'Platform_GB',
      'Platform_SNES', 'User_Score', 'Platform_X360', 'Platform_PC', 'Platform_Wii', 'Publisher_Other', 'Genre_Platform']]

# Apply StandardScalar
scaler = StandardScaler()

# Base models
base_models = [
    ('en', ElasticNet()),
    ('svr', SVR(gamma='scale')),
    ('dtr', DecisionTreeRegressor()),
    ('abr', AdaBoostRegressor()),
    ('rfr', RandomForestRegressor(n_estimators=100)),
    ('etr', ExtraTreesRegressor(n_estimators=100))
]

# Meta-model
meta_model = LinearRegression()

# Stacking Regressor
stacking_model = StackingRegressor(
    estimators = base_models,
    final_estimator = meta_model,
    cv=5
)

# K-fold cross validation (5 splits)
rmseList = []
rsquareLst = []
count    = 1
NUM_SPLITS = 5
kfold = KFold(NUM_SPLITS, shuffle=True)
for train_indices, test_indices in kfold.split(X):
    # Split data
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    # Fit scaler on training fold
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit stacking model
    stacking_model.fit(X_train_scaled, y_train)

    # Predict on test fold
    y_pred = stacking_model.predict(X_test_scaled)

    # RMSE
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmseList.append(rmse)

    # R²
    rsqr = stacking_model.score(X_test_scaled, y_test)
    rsquareLst.append(rsqr)

    print(f"\n*** K-fold {count} ***")
    print("RMSE:", rmse)
    print("R^2:", rsqr)

    count += 1

# Show averages of scores over multiple runs.
print("*********************************************")
print("\nScores for all folds:")
print("*********************************************")
print("RMSE Average :   " + str(np.mean(rmseList)))
print("RMSE SD:         " + str(np.std(rmseList)))
print("RSQ Average :    " + str(np.mean(rsquareLst)))
print("RSQ SD:          " + str(np.std(rsquareLst)))
