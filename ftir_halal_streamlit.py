import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_decomposition import PLSRegression
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

plt.style.use('default')

# Hash password function
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
                    st.session_state.user_role = user_match.iloc[0]['Role'] if 'Role' in user_match.columns else "Standard User"
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
                    st.download_button("\U0001F4C5 Download Registered Users", f, file_name="users.csv", mime="text/csv")

# Main platform logic after successful login
if st.session_state.user_logged_in:
    st.title("FTIR-Based Halal Authentication Platform")

    uploaded_file = st.file_uploader("Upload your dataset (.csv)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df)

        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)

        # PCA with 3 components
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(X_scaled)
        pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])

        show_labels = st.checkbox("Show Sample IDs (PCA)", value=True)
        if 'SampleID' in df.columns and show_labels:
            pc_df['SampleID'] = df['SampleID']
            fig = px.scatter_3d(pc_df, x='PC1', y='PC2', z='PC3', text='SampleID', title="PCA 3D Biplot")
        else:
            fig = px.scatter_3d(pc_df, x='PC1', y='PC2', z='PC3', title="PCA 3D Biplot")
        st.plotly_chart(fig)

        # PLS-DA
        label_col = st.selectbox("Select Label Column (PLS-DA)", df.columns)
        df_clean = df.dropna()
        X = df_clean.drop(label_col, axis=1).select_dtypes(include=[np.number])
        y = df_clean[label_col]
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
        pls = PLSRegression(n_components=3)
        pls.fit(X_train, y_train)
        y_pred_train = pls.predict(X_train)
        y_pred_train_labels = np.round(y_pred_train).astype(int).flatten()
        y_pred_train_labels = np.clip(y_pred_train_labels, 0, len(label_encoder.classes_) - 1)
        y_pred_train_class = label_encoder.inverse_transform(y_pred_train_labels)
        y_train_class = label_encoder.inverse_transform(y_train)
        st.markdown("#### PLS-DA Classification Report (Training Set)")
        report_df = pd.DataFrame(classification_report(y_train_class, y_pred_train_class, output_dict=True)).transpose()
        st.dataframe(report_df.style.set_properties(**{'text-align': 'left'}))

        X_scores = pls.transform(X)
        show_pls_labels = st.checkbox("Show Sample IDs (PLS-DA)", value=False)
        pls_df = pd.DataFrame(X_scores, columns=['PLS1', 'PLS2', 'PLS3'])
        pls_df['Label'] = label_encoder.inverse_transform(y_encoded)
        if 'SampleID' in df.columns and show_pls_labels:
            pls_df['SampleID'] = df['SampleID']
            fig_pls = px.scatter_3d(pls_df, x='PLS1', y='PLS2', z='PLS3', color='Label', text='SampleID', title="PLS-DA Observation Plot")
        else:
            fig_pls = px.scatter_3d(pls_df, x='PLS1', y='PLS2', z='PLS3', color='Label', title="PLS-DA Observation Plot")
        st.plotly_chart(fig_pls)

else:
    st.warning("Please sign in to access the platform.")
