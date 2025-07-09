# File: train_model.py

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("--- Starting Model Training Script ---")

# 2. LOAD DATA
try:
    df = pd.read_csv('Telco-Customer-Churn.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Telco-Customer-Churn.csv' not found. Please download it and place it in the same directory.")
    exit()

# 3. DATA PREPROCESSING & CLEANING
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
print("Data preprocessing complete.")

# 4. FEATURE ENGINEERING & DATA SPLITTING
X = df.drop('Churn', axis=1)
y = df['Churn']

categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. DEFINE THE MODEL PIPELINE
# We will use a Random Forest model
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# 6. TRAIN THE MODEL
# We train on the FULL dataset to have the most robust model for deployment
print("Training the Random Forest model on the full dataset...")
rf_pipeline.fit(X, y)
print("Model training complete.")

# (Optional) You can check accuracy on a test split if you want, but for the final model, we train on all data.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# rf_pipeline.fit(X_train, y_train)
# y_pred = rf_pipeline.predict(X_test)
# print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 7. SAVE THE TRAINED PIPELINE
joblib.dump(rf_pipeline, 'rf_pipeline.joblib')
print("Model pipeline saved successfully as 'rf_pipeline.joblib'.")

print("--- Script Finished ---")
