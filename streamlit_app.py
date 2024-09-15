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
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn import metrics

from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures


import joblib

rating = ["Very Poor","Poor","Fair","Below Average","Average","Above Average",
           "Good","Very Good","Excellent","Very Excellent"]

# Mapping for MSZoning
msZoning_mapping = {
    'Agriculture': 'A',
    'Commercial': 'C',
    'Floating Village Residential': 'FV',
    'Industrial': 'I',
    'Residential High Density': 'RH',
    'Residential Low Density': 'RL',
    'Residential Low Density Park ': 'RP',
    'Residential Medium Density': 'RM'
}

utility_mapping = {
    'All Public Utilities': 'AllPub',
    'Electricity, Gas, and Water (Septic Tank)': 'NoSewr',
    'Electricity and Gas Only': 'NoSeWa',
    'Electricity Only': 'ELO'
}

landSlope_mapping = {
    'Gentle slope': 'Gtl',
    'Moderate Slope': 'Mod',
    'Severe Slope': 'Sev'
}

buildingType_mapping = {
    'Single-family Detached': '1Fam',
    'Two-family Conversion': '2FmCon',
    'Duplex': 'Duplx',
    'Townhouse End Unit': 'TwnhsE',
    'Townhouse Inside Unit': 'TwnhsI'
}

kitchenQual_mapping = {
    'Excellent': 'Ex',
    'Good': 'Gd',
    'Average': 'TA',
    'Fair': 'Fa',
    'Poor': 'Po'
}

saleCondition_mapping = {
    'Normal Sale': 'Normal',
    'Abnormal Sale': 'Abnorml',
    'Adjoining Land Purchase': 'AdjLand',
    'Allocation': 'Alloca',
    'Family': 'Family',
    'Partial': 'Partial'
}


with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('train.csv')
    st.write(df)
           
    st.write('**Statistical Summary of Dataset**')
    summary = df.describe().T
    st.write(summary)

    st.write('**df.head**')
    st.write(df.head())
    
    st.write('**df.shape**')
    st.write(df.shape)

    st.write(df.info())
    st.write(df.describe().T)




#House Price Distribution

plt.figure(figsize=(9, 8))
sns.distplot(df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});

# Step 1: Identify non-numeric columns (optional, for understanding)
st.write(df.dtypes)

# Step 2: Select only numeric columns
numeric_df = df.select_dtypes(include=[float, int])

# Step 3: Plot the heatmap with the correlation matrix of numeric data
plt.figure(figsize=(60, 40))
sns.heatmap(numeric_df.corr(), cmap="RdBu", annot=True, fmt=".2f")  # 'annot' adds correlation coefficients
plt.title("Correlations Between Variables", size=15)
st.write(plt.show())

#select more important columns
important_num_cols = list(numeric_df.corr()["SalePrice"][(numeric_df.corr()["SalePrice"]>0.50) | (numeric_df.corr()["SalePrice"]<-0.50)].index)
cat_cols = ["MSZoning", "Utilities","BldgType","KitchenQual","SaleCondition","LandSlope"]
important_cols = important_num_cols + cat_cols

df = df[important_cols]
st.write(df.info())


df = df.drop(['GarageArea'], axis=1)
st.write(df.info())

important_num_cols.remove("GarageArea")

st.write(df)
#check any missing value
print("Missing Values by Column")
print("-"*30)
print(df.isna().sum())
print("-"*30)
print("TOTAL MISSING VALUES:",df.isna().sum().sum())



sns.pairplot(df.select_dtypes(include=[np.number]))

plt.scatter(x="OverallQual", y="SalePrice", data=df)

plt.scatter(x="YearBuilt", y="SalePrice", data=df)

df.drop(df[(df['YearBuilt'] < 1900) & (df['SalePrice'] > 400000)].index, inplace=True)

plt.scatter(x="YearBuilt", y="SalePrice", data=df)

plt.scatter(x="YearRemodAdd", y="SalePrice", data=df)

df.drop(df[(df['YearRemodAdd'] < 1970) & (df['SalePrice'] > 300000)].index, inplace=True)

plt.scatter(x="TotalBsmtSF", y="SalePrice", data=df)

df.drop(df[(df['TotalBsmtSF'] > 4000)].index, inplace=True)

plt.scatter(x="1stFlrSF", y="SalePrice", data=df)

plt.scatter(x="GrLivArea", y="SalePrice", data=df)

df.drop(df[(df['GrLivArea'] > 4400)].index, inplace=True)

plt.scatter(x="FullBath", y="SalePrice", data=df)

plt.scatter(x="TotRmsAbvGrd", y="SalePrice", data=df)

df.drop(df[(df['TotRmsAbvGrd'] == 14)].index, inplace=True)

plt.scatter(x="GarageCars", y="SalePrice", data=df)

plt.figure(figsize=(10,8))
sns.jointplot(x=df["OverallQual"], y=df["SalePrice"], kind="kde")
sns.jointplot(x=df["YearBuilt"], y=df["SalePrice"], kind="kde")
sns.jointplot(x=df["YearRemodAdd"], y=df["SalePrice"], kind="kde")
sns.jointplot(x=df["TotalBsmtSF"], y=df["SalePrice"], kind="kde")
sns.jointplot(x=df["1stFlrSF"], y=df["SalePrice"], kind="kde")
sns.jointplot(x=df["FullBath"], y=df["SalePrice"], kind="kde")
sns.jointplot(x=df["GrLivArea"], y=df["SalePrice"], kind="kde")
sns.jointplot(x=df["TotRmsAbvGrd"], y=df["SalePrice"], kind="kde")
sns.jointplot(x=df["GarageCars"], y=df["SalePrice"], kind="kde")
plt.show()




X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

#One-Hot encoding
X = pd.get_dummies(X, columns=cat_cols)

important_num_cols.remove("SalePrice")
#Standardization of data
scaler = StandardScaler()
X[important_num_cols] = scaler.fit_transform(X[important_num_cols])

X.head()
st.write(X.head())
st.write(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5)).mean()
    return rmse


def evaluation(y, predictions):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    return mae, mse, rmse, r_squared

models = pd.DataFrame(columns=["Model","MAE","MSE","RMSE","R2 Score","RMSE (Cross-Validation)"])


# Fitting the model and making predictions
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
predictions = lin_reg.predict(X_test)

# Evaluating the model
mae, mse, rmse, r_squared = evaluation(y_test, predictions)
st.write("MAE:", mae)
st.write("MSE:", mse)
st.write("RMSE:", rmse)
st.write("R2 Score:", r_squared)
st.write("-" * 30)


print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)

# Performing cross-validation
rmse_cross_val = rmse_cv(lin_reg)
print("RMSE Cross-Validation:", rmse_cross_val)

# Creating a new row as a DataFrame
new_row = pd.DataFrame({
    "Model": ["LinearRegression"],
    "MAE": [mae],
    "MSE": [mse],
    "RMSE": [rmse],
    "R2 Score": [r_squared],
    "RMSE (Cross-Validation)": [rmse_cross_val]
})

# Concatenating the new row with the existing DataFrame
models = pd.concat([models, new_row], ignore_index=True)

# Display the updated models DataFrame
st.write(models)

# Assuming y_test contains the actual values and predictions contains the predicted values
plt.figure(figsize=(10, 6))

# Scatter plot for Actual Values
plt.scatter(range(len(y_test)), y_test.values, label='Actual', color='b', alpha=0.6)

# Scatter plot for Predicted Values
plt.scatter(range(len(predictions)), predictions, label='Predicted', color='r', alpha=0.6)


# SVR with Grid Search
param_grid = {
    'C': [0.1, 1, 10, 100, 1000, 100000],
    'epsilon': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'poly', 'rbf']
}

svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=1)
grid_search.fit(X_train, y_train)


# Best SVR model and parameters
best_svr = grid_search.best_estimator_
best_params = grid_search.best_params_
svr_predictions = best_svr.predict(X_test)

# Evaluate the SVR model
mae_svr, mse_svr, rmse_svr, r2_svr = evaluation(y_test, svr_predictions)
rmse_cross_val_svr = rmse_cv(best_svr)

st.write("MAE:", mae_svr)
st.write("MSE:", mse_svr)
st.write("RMSE:", rmse_svr)
st.write("R2 Score:", r2_svr)
st.write("-" * 30)

print("SVR (GridSearch) - Best Parameters:", best_params)
print("MAE:", mae_svr)
print("MSE:", mse_svr)
print("RMSE:", rmse_svr)
print("R2 Score:", r2_svr)
print("RMSE Cross-Validation:", rmse_cross_val_svr)
print("-" * 30)

# Create a new row for SVR and update the models DataFrame
new_row_svr = pd.DataFrame({
    "Model": ["SVR (GridSearch)"],
    "MAE": [mae_svr],
    "MSE": [mse_svr],
    "RMSE": [rmse_svr],
    "R2 Score": [r2_svr],
    "RMSE (Cross-Validation)": [rmse_cross_val_svr]
})
models = pd.concat([models, new_row_svr], ignore_index=True)


# Display the updated models DataFrame
st.write(models)

# Plot the Actual vs Predicted values
plt.figure(figsize=(10, 6))

# Scatter plot for Actual Values
plt.scatter(range(len(y_test)), y_test.values, label='Actual', color='b', alpha=0.6)

# Scatter plot for Predicted Values
plt.scatter(range(len(svr_predictions)), svr_predictions, label='Predicted', color='r', alpha=0.6)

# Adding Labels and Title
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Sample Index')
plt.ylabel('SalePrice')
plt.legend()

# Show the plot
plt.show()

# #################################################################################################
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

# st.write("MAE:", best_mae_rfr)
# st.write("MSE:", best_mse_rfr)
# st.write("RMSE:", best_rmse_rfr)
# st.write("R2 Score:", max_r2)
# st.write("-" * 30)

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
# st.write(models)

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
st.write(models)



# # joblib.dump(random_forest, 'random_forest_model.joblib')
# joblib.dump(best_svr, 'svr_model.joblib')
# joblib.dump(lin_reg, 'linear_regression_model.joblib')

# Load the data
with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('train.csv')
    st.write(df)
           
    st.write('**Statistical Summary of Dataset**')
    summary = df.describe().T
    st.write(summary)
           

    # Select more important columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Numeric columns only
    if 'SalePrice' in numeric_df.columns:
        important_num_cols = list(numeric_df.corr()["SalePrice"][
            (numeric_df.corr()["SalePrice"] > 0.50) | (numeric_df.corr()["SalePrice"] < -0.50)
        ].index)
    else:
        important_num_cols = []

    # Categorical columns
    cat_cols = ["MSZoning", "Utilities", "BldgType", "KitchenQual", "SaleCondition", "LandSlope"]
    important_cols = important_num_cols + cat_cols

    df_filtered = df[important_cols]
    st.write("**Filtered Data with Important Columns**")

    df_filtered = df_filtered.drop('GarageArea', axis=1)
    st.write(df_filtered)


with st.expander('Data Visualization'):
    st.write('**Scatter Plot**')
    st.scatter_chart(data=df, x='OverallQual', y='SalePrice')  # Modify as needed

    st.write('**Correlation Heatmap**')
    
    # Filter out non-numeric columns for the heatmap
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if not numeric_df.empty:
        # Generate heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), cmap="RdBu", ax=ax)
        ax.set_title("Correlations Between Variables", size=15)
        st.pyplot(fig)
    else:
        st.write("No numeric columns available for correlation heatmap.")





# Input
with st.sidebar:

    st.header('Input features')
           
    msZoning = st.selectbox('Zoning', list(msZoning_mapping.keys()))
    msZoning_code = msZoning_mapping[msZoning]  # Map to the corresponding code (e.g., "A", "C")
    st.write("Zoning code selected is: ", msZoning_code)

    
    # Utility input with mapping
    utility = st.selectbox('Utility', list(utility_mapping.keys()))
    utility_code = utility_mapping[utility]
    st.write("Utility code selected is: ", utility_code)

    # Land Slope input with mapping
    landSlope = st.selectbox('Land Slope', list(landSlope_mapping.keys()))
    landSlope_code = landSlope_mapping[landSlope]
    st.write("Land Slope code selected is: ", landSlope_code)

    # Building Type input with mapping
    buildingType = st.selectbox('Building Type', list(buildingType_mapping.keys()))
    buildingType_code = buildingType_mapping[buildingType]
    st.write("Building Type code selected is: ", buildingType_code)
           
    overallQuality = st.slider("Rates the overall material and finish of the house", 1, 10, 5)
    st.write("The overall material and finish of the house is : ", rating[overallQuality - 1])

    # Correctly indented date input
    yearBuilt = st.date_input("Original construction date", datetime.date(2019, 7, 6))
    st.write("The original construction date is: ", yearBuilt)
           
    yearRemodAdd = st.date_input("Remodel date", datetime.date(2019, 7, 6))
    st.write("The remodel date date is: ", yearBuilt)

    totalBasmtSF = st.slider("Total square feet of basement area", 0.0, 10000.0, 500.0)
    st.write("Total square feet of basement area is : ", totalBasmtSF, "sqft")

    totalRmsAbvGrd = st.slider("Total rooms above grade (does not include bathrooms)", 1, 20, 10)
    st.write("Total rooms above grade (does not include bathrooms) is : ", totalRmsAbvGrd)


    floorSF = st.slider("First Floor square feet", 0.0, 10000.0, 500.0)
    st.write("First Floor square feet is : ", floorSF, "sqft")

    grLiveArea = st.slider("Above grade (ground) living area square feet", 0.0, 10000.0, 500.0)
    st.write("Above grade (ground) living area square feet is : ", grLiveArea, "sqft")

    fullBath = st.slider("Full bathrooms above grade", 0, 10, 5)
    st.write("Full bathrooms above grade is : ", fullBath)

    # Map kitchen quality input to corresponding code
    kitchenQual = st.selectbox('Kitchen Quality', list(kitchenQual_mapping.keys()))
    kitchenQual_code = kitchenQual_mapping[kitchenQual]  # Map to the corresponding code (e.g., "Ex", "Gd")
    st.write("Kitchen Quality code selected is: ", kitchenQual_code)

    garageCars = st.slider("Size of garage in car capacity", 0, 10, 3)
    st.write("Size of garage in car capacity is : ", grLiveArea)

    # Sale Condition input with mapping
    saleCondition = st.selectbox('Condition of Sale', list(saleCondition_mapping.keys()))
    saleCondition_code = saleCondition_mapping[saleCondition]
    st.write("Sale Condition code selected is: ", saleCondition_code)

    # Corrected data dictionary with valid variable names
    data = {
           'OverallQual': overallQuality,  # Use overallQuality from slider
           'YearBuilt': yearBuilt.year,  # Extract the year from date input
           'YearRemodAdd': yearRemodAdd.year,  # Extract the year from date input
           'TotalBsmtSF': totalBasmtSF,  # Use totalBasmtSF from slider
           'TotRmsAbvGrd': totalRmsAbvGrd,  # Use totalRmsAbvGrd from slider
           '1stFlrSF': floorSF,  # Use floorSF from slider
           'GrLivArea': grLiveArea,  # Use grLiveArea from slider
           'FullBath': fullBath,  # Use fullBath from slider
           'GarageCars': garageCars,  # Use garageCars from slider
           'MSZoning': msZoning_code,  # Use msZoning_code
           'Utilities': utility_code,  # Use utility_code
           'BldgType': buildingType_code,  # Use buildingType_code
           'KitchenQual': kitchenQual_code,  # Use kitchenQual_code from selectbox
           'SaleCondition': saleCondition_code,  # Use saleCondition_code from selectbox
           'LandSlope': landSlope_code,  # Use landSlope_code
    }



# Ensure input_df has the same structure as df_filtered (used in training)
input_df = pd.DataFrame(data, index=[0])
st.write(input_df)
input_data = pd.concat([input_df, df_filtered], axis=0)

important_num_cols.remove("GarageArea")
# Handle categorical variables before numeric scaling
X = pd.get_dummies(input_df, columns=cat_cols)

st.write(X)

# Ensure SalePrice is not in important_num_cols
if 'SalePrice' in important_num_cols:
    important_num_cols.remove("SalePrice")

# Handle the case where the important numeric columns are scaled after dummy encoding
# Check if important_num_cols exist in X
missing_cols = [col for col in important_num_cols if col not in X.columns]

if missing_cols:
    st.write(f"Warning: The following important numeric columns are missing from the dataset after processing: {missing_cols}")

# Standardization of data
scaler = StandardScaler()
# Apply scaler only on numeric columns
X[important_num_cols] = scaler.fit_transform(X[important_num_cols])

# Model selection and prediction
# model_choice = st.selectbox('Select Model', ['Random Forest', 'SVR', 'Linear Regression'])

# Prediction using different models
st.write("## Prediction Results")
if st.button('Predict'):
    # # Linear Regression prediction
    lin_reg_pred = lin_reg.predict(X)
    st.write(f"**Linear Regression Prediction: ${lin_reg_pred[0]:,.2f}**")






