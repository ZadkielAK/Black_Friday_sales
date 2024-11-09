# Import necessary libraries
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set working directory and load datasets
# Update data_dir to the folder containing the CSV files
train_df = pd.read_csv(r"C:\Users\singh\Desktop\DWR1\train.csv")
test_df = pd.read_csv(r"C:\Users\singh\Desktop\DWR1\test.csv")
submission_template = pd.read_csv(r"C:\Users\singh\Desktop\DWR1\sample_submission.csv")

# Display data shapes and preview
print(f"Shape of train data: {train_df.shape}")
print(f"Shape of test data: {test_df.shape}")
print(f"Shape of Submission template: {submission_template.shape}")

# Add User_ID and Product_ID from test to submission template
submission_template['User_ID'] = test_df['User_ID']
submission_template['Product_ID'] = test_df['Product_ID']

# Drop duplicate rows in train data
train_df = train_df.drop_duplicates(keep='first')

# Encode User_ID and Product_ID
le_user = LabelEncoder()
train_df['User_ID'] = le_user.fit_transform(train_df['User_ID'])
test_df['User_ID'] = le_user.transform(test_df['User_ID'])

le_product = LabelEncoder()
train_df['Product_ID'] = le_product.fit_transform(train_df['Product_ID'])

# Handle new Product_IDs in test set
le_product_classes = set(le_product.classes_)
new_product_ids = set(test_df['Product_ID']) - le_product_classes
le_product.classes_ = np.append(le_product.classes_, list(new_product_ids))
test_df['Product_ID'] = le_product.transform(test_df['Product_ID'])

# Drop columns with significant missing values
train_df.drop(columns=['Product_Category_2', 'Product_Category_3'], inplace=True)
test_df.drop(columns=['Product_Category_2', 'Product_Category_3'], inplace=True)

# Reduce categorical variables into fewer categories to simplify model
train_df['Age'] = train_df['Age'].replace({'0-17': '0-25', '18-25': '0-25', '26-35': '26-50', '36-45': '26-50', '46-50': '26-50', '51-55': '51+', '55+': '51+'})
test_df['Age'] = test_df['Age'].replace({'0-17': '0-25', '18-25': '0-25', '26-35': '26-50', '36-45': '26-50', '46-50': '26-50', '51-55': '51+', '55+': '51+'})

# Encode Age column directly to avoid memory issues with get_dummies
le_age = LabelEncoder()
train_df['Age'] = le_age.fit_transform(train_df['Age'])
test_df['Age'] = le_age.transform(test_df['Age'])

# Convert categorical variables with fewer levels into dummy variables
categorical_cols = ['Gender', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']
for col in categorical_cols:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')

train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

# Align columns in train and test data
test_df = test_df.reindex(columns=train_df.columns.drop('Purchase'), fill_value=0)

# Define target and features for model training
X_train = train_df.drop(columns='Purchase')
y_train = train_df['Purchase']

# Split training data for model validation
x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Function to calculate RMSE
def calculate_rmse(model, x, y):
    predictions = model.predict(x)
    return np.sqrt(np.mean((y - predictions) ** 2))

# Linear Regression Model
lm = LinearRegression()
lm.fit(x_train, y_train)
print(f'RMSE Linear Regression: {calculate_rmse(lm, x_val, y_val)}')

# Train on full data and predict for submission
lm.fit(X_train, y_train)
submission_template['Purchase'] = lm.predict(test_df)
submission_template.to_csv('Black_Friday_Sales_Prediction_LR.csv', index=False)

# Random Forest Regressor for better generalization
rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
rf.fit(x_train, y_train)
print(f'RMSE Random Forest: {calculate_rmse(rf, x_val, y_val)}')

# Save models for future use
joblib.dump(lm, "linear_regressor.pkl")
joblib.dump(rf, "random_forest.pkl")

# Visualizing the differences between actual and predicted values for Random Forest
plt.figure(figsize=(16, 6))
y_val_pred = rf.predict(x_val)
plt.scatter(y_val, y_val_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values (Random Forest)")
plt.show()

# Residuals Analysis
residuals = y_val - y_val_pred
plt.figure(figsize=(16, 6))
sns.histplot(residuals, kde=True)
plt.title('Residual Distribution (Random Forest)')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()
print('Mean of Residuals:', np.mean(residuals))
print('Std Deviation of Residuals:', np.std(residuals))
