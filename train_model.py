
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the dataset
df = pd.read_csv('/kaggle/input/loan-approvaldataset/train_u6lujuX_CVtuZ9i (1).csv')

# Show the first few rows of the dataset to check the structure
print(df.head())

# Check for any missing values in the dataset
print(df.isnull().sum())

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(exclude=['number']).columns

# Handle infinite values by replacing them with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Handle missing values:
# Fill missing values in numeric columns with the mean
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Handle categorical columns by filling missing values with the mode
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Ensure numeric columns are actually numeric, coercing any invalid entries
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# If 'Loan_Status' or target variable is categorical, map it to 0 and 1 (if necessary)
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})  # Adjust if 'Loan_Status' is present

# Select features and target variable (adjust column names as needed)
X = df[['LoanAmount', 'ApplicantIncome', 'Credit_History', 'Loan_Amount_Term']]  # Adjust based on dataset
y = df['Loan_Status']  # Adjust if your target variable is different

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict the loan approval status on test data
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

pickle.dump(model, open("loan_model.pkl","wb"))
print("Model saved as loan_model.pkl")
