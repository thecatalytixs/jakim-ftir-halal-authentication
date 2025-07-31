import streamlit as st
import pandas as pd
import hashlib
import os
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
import io

plt.style.use('default')

# Load users data
def load_users():
    if os.path.exists("users.csv"):
        return pd.read_csv("users.csv")
    else:
        return pd.DataFrame(columns=["Email", "Full Name", "Contact Number", "Affiliation", "Password", "Role"])

# Save new user
def save_user(email, full_name, contact, affiliation, password, role):
    users = load_users()
    new_user = pd.DataFrame({
        "Email": [email],
        "Full Name": [full_name],
        "Contact Number": [contact],
        "Affiliation": [affiliation],
        "Password": [hashlib.sha256(password.encode()).hexdigest()],
        "Role": [role]
    })
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv("users.csv", index=False)

# Verify user credentials
def verify_user(email, password):
    users = load_users()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    user_match = users[(users["Email"].str.lower() == email.lower()) & (users["Password"] == password_hash)]
    if not user_match.empty:
        return user_match.iloc[0].to_dict()
    return None

# Initialize session
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_email = ""
    st.session_state.user_role = ""

# Sidebar - Login/Signup
st.sidebar.title("User Access")
access_mode = st.sidebar.radio("", ("Sign In", "Sign Up"))

if access_mode == "Sign In":
    st.sidebar.subheader("Sign In")
    login_email = st.sidebar.text_input("Email")
    login_password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        user = verify_user(login_email, login_password)
        if user:
            st.session_state.authenticated = True
            st.session_state.user_email = user["Email"]
            st.session_state.user_role = user["Role"]
            st.sidebar.success("Login successful!")
        else:
            st.sidebar.error("Invalid email or password.")

elif access_mode == "Sign Up":
    st.sidebar.subheader("Create Account")
    signup_email = st.sidebar.text_input("Email")
    signup_name = st.sidebar.text_input("Full Name")
    signup_contact = st.sidebar.text_input("Contact Number")
    signup_affiliation = st.sidebar.text_input("Affiliation")
    signup_password = st.sidebar.text_input("Password", type="password")

    if signup_email.strip().lower() == "thecatalytixs@gmail.com":
        signup_role = "Admin"
    else:
        signup_role = "Standard User"
    st.sidebar.text(f"Assigned Role: {signup_role}")

    if st.sidebar.button("Register"):
        if signup_email and signup_name and signup_contact and signup_affiliation and signup_password:
            users = load_users()
            if signup_email.lower() in users["Email"].str.lower().values:
                st.sidebar.error("User already exists.")
            else:
                save_user(signup_email, signup_name, signup_contact, signup_affiliation, signup_password, signup_role)
                st.sidebar.success("User registered successfully!")
        else:
            st.sidebar.error("Please fill in all fields.")

# Sign out button
if st.session_state.authenticated:
    if st.sidebar.button("Sign Out"):
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.session_state.user_role = ""
        st.rerun()

# Main App
st.title("FTIR-Based Halal Authentication Platform")

if not st.session_state.authenticated:
    st.write("Welcome to the FTIR Halal Authentication tools. Please sign in to continue.")
else:
    st.write(f"Welcome {st.session_state.user_email}! Upload your dataset to begin analysis.")
