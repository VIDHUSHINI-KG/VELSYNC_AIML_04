# ======================================
# 1. IMPORT LIBRARIES
# ======================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ======================================
# 2. LOAD DATASET
# ======================================

df = pd.read_csv("loan_approval_dataset.csv")


# ======================================
# 3. DATA CLEANING
# ======================================

df.drop(columns=['loan_id'], inplace=True)

# Dataset has no missing values, but safe handling
df.fillna(df.median(numeric_only=True), inplace=True)


# ======================================
# 4. LABEL ENCODING
# ======================================

le = LabelEncoder()

df['education'] = le.fit_transform(df['education'])
df['self_employed'] = le.fit_transform(df['self_employed'])
df['loan_status'] = le.fit_transform(df['loan_status'])


# ======================================
# 5. FEATURES & TARGET
# ======================================

X = df.drop('loan_status', axis=1)
y = df['loan_status']


# ======================================
# 6. TRAIN-TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ======================================
# 7. RANDOM FOREST MODEL (IMPROVED)
# ======================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)


# ======================================
# 8. MODEL EVALUATION
# ======================================

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# ======================================
# 9. FEATURE IMPORTANCE (VERY IMPORTANT)
# ======================================

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop Important Features:\n")
print(feature_importance)

feature_importance.plot(kind='bar', figsize=(10,5))
plt.title("Feature Importance - Loan Approval Prediction")
plt.ylabel("Importance Score")
plt.show()


# ======================================
# 10. CUSTOM PREDICTION FUNCTION
# ======================================

def predict_loan_status(
    no_of_dependents, education, self_employed,
    income_annum, loan_amount, loan_term, cibil_score,
    residential_assets_value, commercial_assets_value,
    luxury_assets_value, bank_asset_value
):
    data = [[
        no_of_dependents, education, self_employed,
        income_annum, loan_amount, loan_term, cibil_score,
        residential_assets_value, commercial_assets_value,
        luxury_assets_value, bank_asset_value
    ]]
    
    prediction = model.predict(data)[0]
    return "Loan Approved" if prediction == 1 else "Loan Rejected"


# ======================================
# 11. EXAMPLE PREDICTION
# ======================================

print(
    predict_loan_status(
        no_of_dependents=1,
        education=1,              # Graduate
        self_employed=0,          # No
        income_annum=700000,
        loan_amount=250000,
        loan_term=360,
        cibil_score=780,
        residential_assets_value=600000,
        commercial_assets_value=0,
        luxury_assets_value=150000,
        bank_asset_value=300000
    )
)
