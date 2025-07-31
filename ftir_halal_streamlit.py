import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_decomposition import PLSRegression
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from supabase import create_client, Client
import os

plt.style.use('default')

# Supabase credentials
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Sign in function
def sign_in(email, password):
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return res
    except Exception as e:
        return None

# Sign up function
def sign_up(email, password):
    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        return res
    except Exception as e:
        return None

# Initialize session
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_email = ""

# Sidebar - Login/Signup
st.sidebar.title("User Access")
access_mode = st.sidebar.radio("Access Mode", ("Sign In", "Sign Up"), label_visibility="collapsed")

if access_mode == "Sign In":
    st.sidebar.subheader("Sign In")
    login_email = st.sidebar.text_input("Email")
    login_password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        user = sign_in(login_email, login_password)
        if user:
            st.session_state.authenticated = True
            st.session_state.user_email = login_email
            st.sidebar.success("Login successful!")
        else:
            st.sidebar.error("Invalid email or password.")

elif access_mode == "Sign Up":
    st.sidebar.subheader("Create Account")
    signup_email = st.sidebar.text_input("Email")
    signup_password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Register"):
        if signup_email and signup_password:
            result = sign_up(signup_email, signup_password)
            if result:
                st.sidebar.success("User registered successfully! Please log in.")
            else:
                st.sidebar.error("Registration failed.")
        else:
            st.sidebar.error("Please fill in all fields.")

# Sign out button
if st.session_state.authenticated:
    if st.sidebar.button("Sign Out"):
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.experimental_rerun()

# Main App
st.title("FTIR-Based Halal Authentication Platform")

if not st.session_state.authenticated:
    st.write("Welcome to the FTIR Halal Authentication tools. Please sign in to continue.")
else:
    st.write(f"Welcome {st.session_state.user_email}! Upload your dataset to begin analysis.")

    uploaded_file = st.file_uploader("Upload your FTIR dataset (CSV format only)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())

        label_col = st.selectbox("Select the label column", df.columns)

        features = df.drop(columns=[label_col])
        labels = df[label_col]

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # PCA Analysis
        st.subheader("PCA Analysis")
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(data=components, columns=["PC1", "PC2"])
        pca_df["Label"] = labels.values

        fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Label", title="PCA Score Plot")
        st.plotly_chart(fig_pca)

        # PLS-DA Analysis
        st.subheader("PLS-DA Analysis")
        pls = PLSRegression(n_components=2)
        pls_fit = pls.fit(X_scaled, y_encoded)
        pls_scores = pls_fit.x_scores_
        pls_df = pd.DataFrame(data=pls_scores, columns=["PLS1", "PLS2"])
        pls_df["Label"] = labels.values

        fig_plsda = px.scatter(pls_df, x="PLS1", y="PLS2", color="Label", title="PLS-DA Score Plot")
        st.plotly_chart(fig_plsda)

        # Logistic Regression Classifier
        st.subheader("Classification using Logistic Regression")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

        st.text("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=label_encoder.classes_, y=label_encoder.classes_,
                                           colorscale='Blues', showscale=True))
        fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig_cm)
