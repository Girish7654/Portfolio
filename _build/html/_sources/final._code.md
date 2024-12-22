## Final Code

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import joblib

# --- Rest of your code ---

data=pd.read_csv("employee.xls")
# Convert Attrition and OverTime to binary
data["Attrition"] = data["Attrition"].map({"Yes": 1, "No": 0})
data["OverTime"] = data["OverTime"].map({"Yes": 1, "No": 0})

# Split the dataset
X = data.drop(columns=["EmployeeID", "Attrition"])
y = data["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Preprocessing Pipeline ---
# Define numeric and categorical columns
num_attribs = [
    "Age", "JobLevel", "JobSatisfaction", "MonthlyIncome", "DailyRate", 
    "HourlyRate", "NumCompaniesWorked", "CompensationPercentSalaryHike", "OverTime"
]
cat_attribs = [
    "Gender", "JobRole", "MaritalStatus", "Department", "EducationField", "BusinessTravel"
]

# Define numeric and categorical pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessing = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
])

# --- SMOTE and Logistic Regression Pipeline ---
pipeline = ImbPipeline([
    ("preprocessing", preprocessing),
    ("smote", SMOTE(random_state=42)),
    ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# --- Save the Model ---
model_path = "smote_with_preprocessing_model.pkl"  # Save in the current directory
joblib.dump(pipeline, model_path)
print(f"Model saved as {model_path}")

# --- Reload and Test the Model ---
reloaded_model = joblib.load(model_path)
reloaded_pred = reloaded_model.predict(X_test)
reloaded_f1 = f1_score(y_test, reloaded_pred)
print("Reloaded Model F1 Score:", reloaded_f1)

import os
print("Current directory:", os.getcwd())
print("File exists:", os.path.exists("smote_with_preprocessing_model.pkl"))