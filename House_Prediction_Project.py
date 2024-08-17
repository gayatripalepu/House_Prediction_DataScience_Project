import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load the dataset
df = pd.read_csv(r'/mnt/data/Housing.csv')
print(df.head())
print(df.info())
print(df.describe())

# Step 2: Check for null values and duplicates
print("\nChecking for null values:")
print(df.isna().sum())
print("\nChecking for duplicate rows:")
print(df.duplicated().any())

# Step 3: Data Visualization
sns.jointplot(data=df, x='price', y='area', kind='hist')
plt.show()

sns.scatterplot(x=df['price'], y=df['area'])
plt.show()

sns.jointplot(data=df, x='price', y='area', kind='scatter')
plt.show()

sns.kdeplot(data=df, x='price', y='area', fill=True)
plt.show()

sns.jointplot(data=df, x='price', y='bedrooms', kind='hist')
plt.show()

sns.jointplot(data=df, x='price', y='bedrooms', kind='scatter')
plt.show()

sns.kdeplot(data=df, x='price', y='bedrooms', fill=True)
plt.show()

sns.pairplot(df)
plt.show()

# Step 4: Encoding categorical variables
col = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
lab_enc = LabelEncoder()

for i in col:
    df[i] = lab_enc.fit_transform(df[i])
print("\nAfter encoding categorical variables:")
print(df.head())

# Step 5: Visualize correlations
plt.figure(figsize=(10,12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Step 6: Standardize the data
std_scaler = StandardScaler()
df_new = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)
print("\nAfter standardization:")
print(df_new.head())

# Step 7: Split the data into features and target
target = df_new['price']
feature = df_new.drop('price', axis=1)

x_train, x_test, y_train, y_test = train_test_split(feature, target, train_size=0.80, random_state=100)

# Step 8: Train and evaluate models

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
y_pred_lin = lin_reg.predict(x_test)
r2_lin = r2_score(y_test, y_pred_lin)
mse_lin = mean_squared_error(y_test, y_pred_lin)
print(f"Linear Regression - R2: {r2_lin}, MSE: {mse_lin}")

sns.regplot(x=y_test, y=y_pred_lin)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Linear Regression)")
plt.show()

# K-Neighbors Regressor
knn = KNeighborsRegressor()
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
r2_knn = r2_score(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
print(f"K-Neighbors Regressor - R2: {r2_knn}, MSE: {mse_knn}")

sns.regplot(x=y_test, y=y_pred_knn)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (K-Neighbors Regressor)")
plt.show()

# Ridge Regression with Hyperparameter Tuning
param_grid = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]}
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')
grid_search.fit(x_train, y_train)
best_ridge = grid_search.best_estimator_
y_pred_ridge = best_ridge.predict(x_test)
r2_ridge = r2_score(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f"Ridge Regression - R2: {r2_ridge}, MSE: {mse_ridge}, Best Alpha: {grid_search.best_params_['alpha']}")

sns.regplot(x=y_test, y=y_pred_ridge)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Ridge Regression)")
plt.show()

# Step 9: Model Comparison
data = {
    'Model': ['Linear Regression', 'KNeighbors Regressor', 'Ridge Regression'],
    'R2_Score': [r2_lin, r2_knn, r2_ridge],
    'MSE': [mse_lin, mse_knn, mse_ridge]
}
df_comparison = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2_Score', data=df_comparison)
plt.title("Model Comparison (R2 Score)")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MSE', data=df_comparison)
plt.title("Model Comparison (Mean Squared Error)")
plt.show()
