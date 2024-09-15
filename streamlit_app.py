import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import sklearn # import the module so you can use it
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures


import joblib

with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('train.csv')
    st.write(df)
           
    st.write('**Statistical Summary of Dataset**')
    summary = df.describe().T
    st.write(summary)
  
    st.write(df.head())
    
df.shape
    

# # Function to create scrollable table within a small window
# def create_scrollable_table(df, table_id, title):
#     html = f'<h3>{title}</h3>'
#     html += f'<div id="{table_id}" style="height:200px; overflow:auto;">'
#     html += df.to_html()
#     html += '</div>'
#     return html

# # Summary statistics for numerical features
# numerical_features = df.select_dtypes(include=[np.number])
# summary_stats = numerical_features.describe().T
# html_numerical = create_scrollable_table(summary_stats, 'numerical_features', 'Summary statistics for numerical features')

# display(HTML(html_numerical))

# # Summary statistics for categorical features
# categorical_features = df.select_dtypes(include=[object])
# cat_summary_stats = categorical_features.describe().T
# html_categorical = create_scrollable_table(cat_summary_stats, 'categorical_features', 'Summary statistics for categorical features')

# display(HTML(html_categorical ))

# #House Price Distribution
# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(9, 8))
# sns.distplot(df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});

# # Step 1: Identify non-numeric columns (optional, for understanding)
# print(df.dtypes)

# # Step 2: Select only numeric columns
# numeric_df = df.select_dtypes(include=[float, int])

# # Step 3: Plot the heatmap with the correlation matrix of numeric data
# plt.figure(figsize=(60, 40))
# sns.heatmap(numeric_df.corr(), cmap="RdBu", annot=True, fmt=".2f")  # 'annot' adds correlation coefficients
# plt.title("Correlations Between Variables", size=15)
# plt.show()

# #select more important columns
# important_num_cols = list(numeric_df.corr()["SalePrice"][(numeric_df.corr()["SalePrice"]>0.50) | (numeric_df.corr()["SalePrice"]<-0.50)].index)
# cat_cols = ["MSZoning", "Utilities","BldgType","KitchenQual","SaleCondition","LandSlope"]
# important_cols = important_num_cols + cat_cols

# df = df[important_cols]
# df.info()

# correlation_matrix = df.corr(numeric_only=True)
# plt.figure(figsize=(20,12))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# df = df.drop(['GarageArea'], axis=1)
# df.info()

# important_num_cols.remove("GarageArea")

# #check any missing value
# print("Missing Values by Column")
# print("-"*30)
# print(df.isna().sum())
# print("-"*30)
# print("TOTAL MISSING VALUES:",df.isna().sum().sum())

# sns.pairplot(df.select_dtypes(include=[np.number]))

# plt.scatter(x="OverallQual", y="SalePrice", data=df)

# plt.scatter(x="YearBuilt", y="SalePrice", data=df)

# df.drop(df[(df['YearBuilt'] < 1900) & (df['SalePrice'] > 400000)].index, inplace=True)

# plt.scatter(x="YearBuilt", y="SalePrice", data=df)

# plt.scatter(x="YearRemodAdd", y="SalePrice", data=df)

# df.drop(df[(df['YearRemodAdd'] < 1970) & (df['SalePrice'] > 300000)].index, inplace=True)

# plt.scatter(x="TotalBsmtSF", y="SalePrice", data=df)

# df.drop(df[(df['TotalBsmtSF'] > 4000)].index, inplace=True)

# plt.scatter(x="1stFlrSF", y="SalePrice", data=df)

# plt.scatter(x="GrLivArea", y="SalePrice", data=df)

# df.drop(df[(df['GrLivArea'] > 4400)].index, inplace=True)

# plt.scatter(x="FullBath", y="SalePrice", data=df)

# plt.scatter(x="TotRmsAbvGrd", y="SalePrice", data=df)

# df.drop(df[(df['TotRmsAbvGrd'] == 14)].index, inplace=True)

# plt.scatter(x="GarageCars", y="SalePrice", data=df)

# plt.figure(figsize=(10,8))
# sns.jointplot(x=df["OverallQual"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["YearBuilt"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["YearRemodAdd"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["TotalBsmtSF"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["1stFlrSF"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["FullBath"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["GrLivArea"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["TotRmsAbvGrd"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["GarageCars"], y=df["SalePrice"], kind="kde")
# plt.show()

# X = df.drop("SalePrice", axis=1)
# y = df["SalePrice"]

# #One-Hot encoding
# X = pd.get_dummies(X, columns=cat_cols)

# important_num_cols.remove("SalePrice")
# #Standardization of data
# scaler = StandardScaler()
# X[important_num_cols] = scaler.fit_transform(X[important_num_cols])

# X.head()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# def rmse_cv(model):
#     rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5)).mean()
#     return rmse


# def evaluation(y, predictions):
#     mae = mean_absolute_error(y, predictions)
#     mse = mean_squared_error(y, predictions)
#     rmse = np.sqrt(mean_squared_error(y, predictions))
#     r_squared = r2_score(y, predictions)
#     return mae, mse, rmse, r_squared

# models = pd.DataFrame(columns=["Model","MAE","MSE","RMSE","R2 Score","RMSE (Cross-Validation)"])

# import pandas as pd
# from sklearn.linear_model import LinearRegression

# # Fitting the model and making predictions
# lin_reg = LinearRegression()
# lin_reg.fit(X_train, y_train)
# predictions = lin_reg.predict(X_test)

# # Evaluating the model
# mae, mse, rmse, r_squared = evaluation(y_test, predictions)
# print("MAE:", mae)
# print("MSE:", mse)
# print("RMSE:", rmse)
# print("R2 Score:", r_squared)
# print("-" * 30)

# # Performing cross-validation
# rmse_cross_val = rmse_cv(lin_reg)
# print("RMSE Cross-Validation:", rmse_cross_val)

# # Creating a new row as a DataFrame
# new_row = pd.DataFrame({
#     "Model": ["LinearRegression"],
#     "MAE": [mae],
#     "MSE": [mse],
#     "RMSE": [rmse],
#     "R2 Score": [r_squared],
#     "RMSE (Cross-Validation)": [rmse_cross_val]
# })

# # Concatenating the new row with the existing DataFrame
# models = pd.concat([models, new_row], ignore_index=True)

# # Display the updated models DataFrame
# print(models)

# # Assuming y_test contains the actual values and predictions contains the predicted values
# plt.figure(figsize=(10, 6))

# # Scatter plot for Actual Values
# plt.scatter(range(len(y_test)), y_test.values, label='Actual', color='b', alpha=0.6)

# # Scatter plot for Predicted Values
# plt.scatter(range(len(predictions)), predictions, label='Predicted', color='r', alpha=0.6)

# # Adding Labels and Title
# plt.title('Actual vs Predicted House Prices')
# plt.xlabel('Sample Index')
# plt.ylabel('SalePrice')
# plt.legend()

# # Show the plot
# plt.show()

# from sklearn.model_selection import GridSearchCV
# from scipy import stats
# from sklearn import metrics

# # SVR with Grid Search
# param_grid = {
#     'C': [0.1, 1, 10, 100, 1000, 100000],
#     'epsilon': [0.001, 0.01, 0.1, 1],
#     'kernel': ['linear', 'poly', 'rbf']
# }

# svr = SVR()
# grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=1)
# grid_search.fit(X_train, y_train)


# # Best SVR model and parameters
# best_svr = grid_search.best_estimator_
# best_params = grid_search.best_params_
# svr_predictions = best_svr.predict(X_test)

# # Evaluate the SVR model
# mae_svr, mse_svr, rmse_svr, r2_svr = evaluation(y_test, svr_predictions)
# rmse_cross_val_svr = rmse_cv(best_svr)

# print("SVR (GridSearch) - Best Parameters:", best_params)
# print("MAE:", mae_svr)
# print("MSE:", mse_svr)
# print("RMSE:", rmse_svr)
# print("R2 Score:", r2_svr)
# print("RMSE Cross-Validation:", rmse_cross_val_svr)
# print("-" * 30)

# # Create a new row for SVR and update the models DataFrame
# new_row_svr = pd.DataFrame({
#     "Model": ["SVR (GridSearch)"],
#     "MAE": [mae_svr],
#     "MSE": [mse_svr],
#     "RMSE": [rmse_svr],
#     "R2 Score": [r2_svr],
#     "RMSE (Cross-Validation)": [rmse_cross_val_svr]
# })
# models = pd.concat([models, new_row_svr], ignore_index=True)


# # Display the updated models DataFrame
# print(models)

# # Plot the Actual vs Predicted values
# plt.figure(figsize=(10, 6))

# # Scatter plot for Actual Values
# plt.scatter(range(len(y_test)), y_test.values, label='Actual', color='b', alpha=0.6)

# # Scatter plot for Predicted Values
# plt.scatter(range(len(svr_predictions)), svr_predictions, label='Predicted', color='r', alpha=0.6)

# # Adding Labels and Title
# plt.title('Actual vs Predicted House Prices')
# plt.xlabel('Sample Index')
# plt.ylabel('SalePrice')
# plt.legend()

# # Show the plot
# plt.show()

# # Random Forest Regressor
# max_r2 = 0
# for n_trees in range(64, 129):
#     random_forest = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1)
#     random_forest.fit(X_train, y_train)

#     rfr_predictions = random_forest.predict(X_test)
#     mae_rfr, mse_rfr, rmse_rfr, r2_rfr = evaluation(y_test, rfr_predictions)
#     rmse_cross_val_rfr = rmse_cv(random_forest)

#     print('For a Random Forest with', n_trees, 'trees in total:')
#     print('MAE: %0.5f'%mae_rfr)
#     print('MSE: %0.5f'%mse_rfr)
#     print('RMSE: %0.5f'%rmse_rfr)
#     print('R2 Score: %0.5f' %r2_rfr)
#     print("RMSE Cross-Validation: %0.5f"%rmse_cross_val_rfr)
#     print('--------------------------------------')

#     if r2_rfr > max_r2:
#         max_r2 = r2_rfr
#         best_mae_rfr = mae_rfr
#         best_mse_rfr = mse_rfr
#         best_rmse_rfr = rmse_rfr
#         best_rmse_cv_rfr = rmse_cross_val_rfr
#         best_n_trees = n_trees

# print(f'Highest R2 Score for Random Forest: {max_r2} at {best_n_trees} trees')
# print("MAE:", best_mae_rfr)
# print("MSE:", best_mse_rfr)
# print("RMSE:", best_rmse_rfr)
# print("R2 Score:", max_r2)
# print("RMSE Cross-Validation:", best_rmse_cv_rfr)
# print("-" * 30)

# # Add RandomForestRegressor to the models DataFrame
# new_row_rfr = pd.DataFrame({
#     "Model": ["RandomForestRegressor"],
#     "MAE": [best_mae_rfr],
#     "MSE": [best_mse_rfr],
#     "RMSE": [best_rmse_rfr],
#     "R2 Score": [max_r2],
#     "RMSE (Cross-Validation)": [best_rmse_cv_rfr]
# })
# models = pd.concat([models, new_row_rfr], ignore_index=True)

# # Display the updated models DataFrame
# print(models)

# # Assuming y_test contains the actual values and predictions contains the predicted values
# plt.figure(figsize=(10, 6))

# # Scatter plot for Actual Values
# plt.scatter(range(len(y_test)), y_test.values, label='Actual', color='b', alpha=0.6)

# # Scatter plot for Predicted Values
# plt.scatter(range(len(rfr_predictions)), rfr_predictions, label='Predicted', color='r', alpha=0.6)

# # Adding Labels and Title
# plt.title('Actual vs Predicted House Prices')
# plt.xlabel('Sample Index')
# plt.ylabel('SalePrice')
# plt.legend()

# # Show the plot
# plt.show()

# models.sort_values(by="RMSE (Cross-Validation)")

# plt.figure(figsize=(12,8))
# sns.barplot(x=models["Model"], y=models["RMSE (Cross-Validation)"])
# plt.title("Models' RMSE Scores (Cross-Validated)", size=15)
# plt.xticks(rotation=30, size=12)
# plt.show()

# plt.figure(figsize=(10,6))
# sns.barplot(x=models["Model"], y=models["R2 Score"])
# plt.title("Models' R2 Scores", size=15)
# plt.xticks(rotation=30, size=12)
# plt.show()

# pip install --upgrade scikit-learn

# joblib.dump(random_forest, 'random_forest_model.joblib')
# joblib.dump(best_svr, 'svr_model.joblib')
# joblib.dump(lin_reg, 'linear_regression_model.joblib')

