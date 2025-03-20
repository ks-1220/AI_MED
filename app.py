import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, r2_score, roc_auc_score

# Streamlit App Title
st.set_page_config(page_title="AI MedX Predictor", layout="wide")
st.sidebar.title("Navigation")

# Create sidebar navigation
page = st.sidebar.radio(
    "Select Prediction", 
    ["Enrollment Duration (PS-2)", "Trial Completion (PS-3)", "Recruitment Rate (PS-4)"]
)

# Load dataset
data_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])
if data_file is not None:
    data = pd.read_csv(data_file)
    
    # Rename NCT Number column for consistency
    data.rename(columns={'NCT Number': 'nct_id'}, inplace=True)

    # Define features and target
    features = ['nct_id', 'Study Title', 'Study Status', 'Conditions', 'Interventions',
                'Primary Outcome Measures', 'Sponsor', 'Funder Type', 'Sex', 'Phases',
                'Age', 'Enrollment', 'Study Type', 'Study Design', 'Locations', 'Study Recruitment Rate']
    
    numerical_features = ['Study Recruitment Rate', 'Enrollment']
    categorical_features = ['Study Status', 'Phases', 'Age', 'Sponsor', 'Funder Type', 'Study Type', 'Sex']
    text_features = ['Study Title', 'Conditions', 'Interventions', 'Primary Outcome Measures']
    
    # Handle missing values
    data[numerical_features] = data[numerical_features].fillna(0)
    data[categorical_features] = data[categorical_features].fillna('')
    data[text_features] = data[text_features].fillna('')
    
    # Preprocess Data
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown='ignore')
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)

    X_numerical = scaler.fit_transform(data[numerical_features])
    X_categorical = encoder.fit_transform(data[categorical_features])
    tfidf_matrix = vectorizer.fit_transform(data[text_features].apply(lambda row: ' '.join(row), axis=1))

    X_final = hstack([X_numerical, X_categorical, tfidf_matrix])

    # Define Targets (PS-2, PS-3, PS-4)
    y = data[['Enrollment', 'Study Status', 'Study Recruitment Rate']].copy()
    y['Study Status'] = y['Study Status'].apply(lambda x: 1 if x == 'COMPLETED' else 0)

    # Apply SMOTE for classification
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled_classification = smote.fit_resample(X_final.toarray(), y['Study Status'])
    
    # Create a new DataFrame with resampled classification target
    y_resampled = pd.DataFrame(y_resampled_classification, columns=['Study Status'])
    y_resampled['Enrollment'] = np.repeat(y['Enrollment'].values, len(y_resampled) // len(y) + 1)[:len(y_resampled)]
    y_resampled['Study Recruitment Rate'] = np.repeat(y['Study Recruitment Rate'].values, len(y_resampled) // len(y) + 1)[:len(y_resampled)]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train Models
    regressor = MultiOutputRegressor(XGBRegressor(n_estimators=200, learning_rate=0.03, random_state=42))
    classifier = XGBClassifier(n_estimators=200, learning_rate=0.03, random_state=42)

    # Train Regression Model (PS-2 & PS-4)
    regressor.fit(X_train, y_train[['Enrollment', 'Study Recruitment Rate']])

    # Train Classification Model (PS-3)
    classifier.fit(X_train, y_train['Study Status'])

    # Predictions
    y_pred_regression = regressor.predict(X_test)
    y_pred_classification = classifier.predict(X_test).reshape(-1, 1)
    y_pred = np.hstack([y_pred_regression[:, 0].reshape(-1, 1), y_pred_classification, y_pred_regression[:, 1].reshape(-1, 1)])

    # Display Predictions Based on Page Selection
    if page == "Enrollment Duration (PS-2)":
        st.title("Predict Enrollment Duration")
        st.write(pd.DataFrame({"Enrollment Prediction": y_pred[:, 0]}))

    elif page == "Trial Completion (PS-3)":
        st.title("Predict Trial Completion")
        st.write(pd.DataFrame({"Study Status Prediction": y_pred[:, 1]}))

    elif page == "Recruitment Rate (PS-4)":
        st.title("Predict Recruitment Rate")
        st.write(pd.DataFrame({"Recruitment Rate Prediction": y_pred[:, 2]}))

    # Feature Importance (SHAP)
    st.subheader("Feature Importance using SHAP")
    explainer = shap.Explainer(regressor.estimators_[0], X_test)
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test)
    st.pyplot(plt)
