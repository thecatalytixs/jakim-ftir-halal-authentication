import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_decomposition import PLSRegression
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import os

# Force matplotlib to use light theme
plt.style.use('default')

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

st.set_page_config(page_title="FTIR-Based Halal Authentication", layout="wide")

# Sidebar for user authentication
with st.sidebar:
    auth_option = st.radio("User Access", ["Sign In", "Sign Up"])

    file_path = "users.csv"
    if 'user_logged_in' not in st.session_state:
        st.session_state.user_logged_in = False
        st.session_state.user_email = ""
        st.session_state.user_role = ""

    if auth_option == "Sign Up":
        st.header("Create Account")
        signup_email = st.text_input("Email")
        signup_name = st.text_input("Full Name")
        signup_contact = st.text_input("Contact Number")
        signup_affiliation = st.text_input("Affiliation")
        signup_password = st.text_input("Password", type="password")
        signup_role = st.selectbox("Select Role", ["Standard User", "Admin"])
        if st.button("Register"):
            if not signup_email or not signup_name or not signup_contact or not signup_affiliation or not signup_password:
                st.warning("Please fill in all fields to register.")
            else:
                user_data = {
                    "Email": signup_email,
                    "Full Name": signup_name,
                    "Contact Number": signup_contact,
                    "Affiliation": signup_affiliation,
                    "Password": hash_password(signup_password),
                    "Role": signup_role
                }
                if os.path.exists(file_path):
                    existing_users = pd.read_csv(file_path)
                    if signup_email in existing_users['Email'].values:
                        st.error("Email already registered. Please sign in.")
                    else:
                        existing_users = pd.concat([existing_users, pd.DataFrame([user_data])], ignore_index=True)
                        existing_users.to_csv(file_path, index=False)
                        st.success("Registration successful! Please sign in to continue.")
                else:
                    pd.DataFrame([user_data]).to_csv(file_path, index=False)
                    st.success("Registration successful! Please sign in to continue.")

    elif auth_option == "Sign In":
        st.header("Sign In")
        signin_email = st.text_input("Email")
        signin_password = st.text_input("Password", type="password")
        if st.button("Login"):
            if os.path.exists(file_path):
                users = pd.read_csv(file_path)
                hashed_input_password = hash_password(signin_password)
                user_match = users[(users['Email'] == signin_email) & (users['Password'] == hashed_input_password)]
                if not user_match.empty:
                    st.success("Login successful!")
                    st.session_state.user_logged_in = True
                    st.session_state.user_email = signin_email
                    if 'Role' in user_match.columns:
                        st.session_state.user_role = user_match.iloc[0].get('Role', 'Standard User')
                    else:
                        st.error("User role column is missing. Please check the users.csv file structure.")
                        st.stop()
                else:
                    st.error("Invalid email or password.")
            else:
                st.warning("No registered users found. Please sign up first.")

    if st.session_state.user_logged_in:
        if st.button("Sign Out"):
            st.session_state.user_logged_in = False
            st.session_state.user_email = ""
            st.session_state.user_role = ""
            st.experimental_rerun()

        if st.session_state.user_role == "Admin":
            st.markdown("### Developer Tools")
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    st.download_button("📥 Download Registered Users", f, file_name="users.csv", mime="text/csv")

# Main platform logic after successful login
if st.session_state.user_logged_in:
    st.title("FTIR-Based Halal Authentication Platform")

    # Optional: Show admin tools again (if needed)
    if st.session_state.user_role == "Admin":
        st.subheader("🔧 Admin Access")
        # Additional admin features can go here

    # Main content for both Admin and Standard User
    st.write("Welcome to the FTIR Halal Authentication tools. Upload your dataset to begin analysis.")
    # TODO: Insert analysis modules like PCA, PLSDA, etc. here

else:
    st.warning("Please sign in to access the platform.")
