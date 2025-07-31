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
        st.experimental_rerun()

# Main App
st.title("FTIR-Based Halal Authentication Platform")

if not st.session_state.authenticated:
    st.write("Welcome to the FTIR Halal Authentication tools. Please sign in to continue.")
else:
    st.write(f"Welcome {st.session_state.user_email}! Upload your dataset to begin analysis.")

    if st.session_state.user_role in ["Admin", "Standard User"]:
        uploaded_file = st.file_uploader("Upload your FTIR dataset (CSV format only)", type=["csv"])

        @st.cache_data
        def load_data(uploaded_file):
            if uploaded_file is not None:
                return pd.read_csv(uploaded_file)
            else:
                return pd.DataFrame()

        df = load_data(uploaded_file)
        if not df.empty:
            st.subheader("1. Preview of Uploaded Dataset")
            st.dataframe(df.head())

            X = df.drop(columns=["SampleID", "Class"])
            y = df["Class"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            st.subheader("2. Principal Component Analysis (PCA)")
            pca = PCA(n_components=3)
            pca_scores = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(data=pca_scores, columns=["PC1", "PC2", "PC3"])
            pca_df["Class"] = y.values
            pca_df["SampleID"] = df["SampleID"].values

            show_labels = st.checkbox("Show SampleID labels on PCA plot")
            if show_labels:
                fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="Class", text="SampleID", title="PCA Score Plot (3D)")
                fig.update_traces(textposition='top center')
            else:
                fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="Class", title="PCA Score Plot (3D)")

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("3. Variable Plot (PCA Loadings)")
            loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2", "PC3"], index=X.columns)
            fig_loadings = px.scatter_3d(loadings.reset_index(), x="PC1", y="PC2", z="PC3", text="index")
            fig_loadings.update_layout(title="PCA Loadings Plot (PC1 vs PC2 vs PC3)", scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
            st.plotly_chart(fig_loadings, use_container_width=True)

            st.subheader("4. PCA Biplot")
            fig_biplot = go.Figure()
            for label in pca_df["Class"].unique():
                subset = pca_df[pca_df["Class"] == label]
                fig_biplot.add_trace(go.Scatter3d(x=subset["PC1"], y=subset["PC2"], z=subset["PC3"],
                                                  mode='markers+text' if show_labels else 'markers',
                                                  text=subset["SampleID"] if show_labels else None,
                                                  name=label))

            for i in range(loadings.shape[0]):
                fig_biplot.add_trace(go.Scatter3d(x=[0, loadings.iloc[i, 0]*3], y=[0, loadings.iloc[i, 1]*3], z=[0, loadings.iloc[i, 2]*3],
                                                  mode='lines+text',
                                                  text=["", loadings.index[i]],
                                                  name=loadings.index[i],
                                                  line=dict(color='black', width=2)))

            fig_biplot.update_layout(title="PCA Biplot (PC1 vs PC2 vs PC3)",
                                     scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
            st.plotly_chart(fig_biplot, use_container_width=True)

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

            y_pred_pls = pls.predict(X_scaled)
            y_pred_labels = np.round(y_pred_pls).astype(int).flatten()
            y_pred_labels = np.clip(y_pred_labels, 0, len(label_encoder.classes_) - 1)
            y_pred_classes = label_encoder.inverse_transform(y_pred_labels)

            report_dict = classification_report(y, y_pred_classes, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose().round(2)
            st.markdown("**Classification Report (Training Set via PLS-DA):**")
            st.dataframe(report_df, use_container_width=True)

            pls_conf_matrix = confusion_matrix(y, y_pred_classes)
            st.markdown("**Confusion Matrix (Training Set via PLS-DA):**")
            st.dataframe(pd.DataFrame(pls_conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_))

            T = pls.x_scores_
            W = pls.x_weights_
            Q = pls.y_loadings_
            p, h = W.shape
            SStotal = np.sum(np.square(T), axis=0) * np.square(Q).flatten()
            vip = np.sqrt(p * np.sum((W**2) * SStotal.reshape(1, -1), axis=1) / np.sum(SStotal))

            vip_df = pd.DataFrame({'Variable': X.columns, 'VIP_Score': vip})
            vip_df = vip_df.sort_values(by='VIP_Score', ascending=False)

            fig_vip = px.bar(vip_df.head(20), x='Variable', y='VIP_Score', title='Top 20 VIP Scores')
            st.plotly_chart(fig_vip, use_container_width=True)
            st.dataframe(vip_df)

            st.subheader("6. Logistic Regression Classification (Test Set)")
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)

            report_dict_test = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, output_dict=True)
            report_df_test = pd.DataFrame(report_dict_test).transpose().round(2)
            st.markdown("**Classification Report (Test Set):**")
            st.dataframe(report_df_test, use_container_width=True)

            conf_matrix_test = confusion_matrix(y_test, y_pred_test)
            st.markdown("**Confusion Matrix (Test Set):**")
            st.dataframe(pd.DataFrame(conf_matrix_test, index=label_encoder.classes_, columns=label_encoder.classes_))

        else:
            st.warning("Please upload a valid dataset to start the analysis.")
    else:
        st.warning("You do not have permission to access this feature.")
