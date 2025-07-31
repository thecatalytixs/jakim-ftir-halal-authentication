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
                    st.session_state.user_role = user_match.iloc[0]['Role']
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

    if st.session_state.user_logged_in and st.session_state.user_role == "Admin":
        st.markdown("### Developer Tools")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                st.download_button("📥 Download Registered Users", f, file_name="users.csv", mime="text/csv")

if not st.session_state.user_logged_in:
    st.warning("Please sign in to access the platform.")
    st.stop()

st.title("FTIR-Based Halal Authentication Platform")

# File uploader
uploaded_file = st.file_uploader("Upload your FTIR dataset (CSV format only)", type=["csv"])

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_csv("simulated_ftir_data.csv")  # fallback demo file

df = load_data(uploaded_file)
st.subheader("1. Preview of Uploaded Dataset")
st.dataframe(df.head())

# Extract features and class labels
X = df.drop(columns=["SampleID", "Class"])
y = df["Class"]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# PCA
st.subheader("2. Principal Component Analysis (PCA)")
pca = PCA(n_components=3)
pca_scores = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=pca_scores, columns=["PC1", "PC2", "PC3"])
pca_df["Class"] = y.values
pca_df["SampleID"] = df["SampleID"].values

# Option to toggle SampleID labels
show_labels = st.checkbox("Show SampleID labels on PCA plot")
if show_labels:
    fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="Class", text="SampleID", title="PCA Score Plot (3D)")
    fig.update_traces(textposition='top center')
else:
    fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="Class", title="PCA Score Plot (3D)")

st.plotly_chart(fig, use_container_width=True)

# PCA Loadings (Variable Plot)
st.subheader("3. Variable Plot (PCA Loadings)")
loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2", "PC3"], index=X.columns)
fig_loadings = px.scatter_3d(loadings.reset_index(), x="PC1", y="PC2", z="PC3", text="index")
fig_loadings.update_layout(title="PCA Loadings Plot (PC1 vs PC2 vs PC3)", scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
st.plotly_chart(fig_loadings, use_container_width=True)

# PCA Biplot
st.subheader("4. PCA Biplot")
fig_biplot = go.Figure()

# Add scores
for label in pca_df["Class"].unique():
    subset = pca_df[pca_df["Class"] == label]
    fig_biplot.add_trace(go.Scatter3d(x=subset["PC1"], y=subset["PC2"], z=subset["PC3"],
                                      mode='markers+text' if show_labels else 'markers',
                                      text=subset["SampleID"] if show_labels else None,
                                      name=label))

# Add loadings
for i in range(loadings.shape[0]):
    fig_biplot.add_trace(go.Scatter3d(x=[0, loadings.iloc[i, 0]*3], y=[0, loadings.iloc[i, 1]*3], z=[0, loadings.iloc[i, 2]*3],
                                      mode='lines+text',
                                      text=["", loadings.index[i]],
                                      name=loadings.index[i],
                                      line=dict(color='black', width=2)))

fig_biplot.update_layout(title="PCA Biplot (PC1 vs PC2 vs PC3)",
                         scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
st.plotly_chart(fig_biplot, use_container_width=True)

# PLS-DA Scores Plot
st.subheader("5. PLS-DA Observation Plot")
pls = PLSRegression(n_components=3)
pls.fit(X_scaled, y_encoded)
pls_scores = pls.x_scores_
pls_df = pd.DataFrame(pls_scores, columns=["PLS1", "PLS2", "PLS3"])
pls_df["Class"] = y.values
pls_df["SampleID"] = df["SampleID"].values

show_pls_labels = st.checkbox("Show SampleID labels on PLS-DA plot")
if show_pls_labels:
    fig_pls = px.scatter_3d(pls_df, x="PLS1", y="PLS2", z="PLS3", color="Class", text="SampleID", title="PLS-DA Observation Plot (3D)")
    fig_pls.update_traces(textposition='top center')
else:
    fig_pls = px.scatter_3d(pls_df, x="PLS1", y="PLS2", z="PLS3", color="Class", title="PLS-DA Observation Plot (3D)")

st.plotly_chart(fig_pls, use_container_width=True)
