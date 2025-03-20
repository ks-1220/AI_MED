import streamlit as st
import pandas as pd
import numpy as np
import joblib
#import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import hstack
from xgboost import XGBRegressor, XGBClassifier
from sklearn.multioutput import MultiOutputRegressor

def load_model():
    return joblib.load("model.pkl")

def preprocess_input(data, scaler, encoder, vectorizer):
    X_numerical = scaler.transform(data[['Study Recruitment Rate', 'Enrollment']])
    X_categorical = encoder.transform(data[['Study Status', 'Phases', 'Age', 'Sponsor', 'Funder Type', 'Study Type', 'Sex']])
    tfidf_matrix = vectorizer.transform(data[['Study Title', 'Conditions', 'Interventions', 'Primary Outcome Measures']].apply(lambda row: ' '.join(row), axis=1))
    return hstack([X_numerical, X_categorical, tfidf_matrix])

st.set_page_config(page_title="AI MedX Predictor", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Prediction", ["Enrollment Duration (PS-2)", "Trial Completion (PS-3)", "Recruitment Rate (PS-4)"])

model = load_model()
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
vectorizer = joblib.load("vectorizer.pkl")

data_input = st.file_uploader("Upload test dataset (CSV)", type=["csv"])
if data_input is not None:
    test_data = pd.read_csv(data_input)
    X_test = preprocess_input(test_data, scaler, encoder, vectorizer)
    y_pred = model.predict(X_test)
    y_pred[:, 1] = (y_pred[:, 1] > 0.5).astype(int)

    if page == "Enrollment Duration (PS-2)":
        st.title("Predict Enrollment Duration")
        st.write(pd.DataFrame({"Enrollment Prediction": y_pred[:, 0]}))
    elif page == "Trial Completion (PS-3)":
        st.title("Predict Trial Completion")
        st.write(pd.DataFrame({"Study Status Prediction": y_pred[:, 1]}))
    elif page == "Recruitment Rate (PS-4)":
        st.title("Predict Recruitment Rate")
        st.write(pd.DataFrame({"Recruitment Rate Prediction": y_pred[:, 2]}))

    # SHAP Explanation
    explainer = shap.Explainer(model.regression_model.estimators_[0], X_test)
    shap_values = explainer.shap_values(X_test)
    st.subheader("Feature Importance using SHAP")
    plt.figure()
    shap.summary_plot(shap_values, X_test)
    st.pyplot(plt)
