import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px

st.set_page_config(page_title="FTIR-Based Halal Authentication", layout="wide")
st.title("FTIR-Based Halal Authentication Platform")

# Load example dataset
@st.cache_data
def load_data():
    df = pd.read_csv("simulated_ftir_data.csv")  # replace with actual path or file uploader
    return df

df = load_data()
st.subheader("1. Preview of Uploaded Dataset")
st.dataframe(df.head())

# Extract features and class labels
X = df.drop(columns=["SampleID", "Class"])
y = df["Class"]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
st.subheader("2. Principal Component Analysis (PCA)")
pca = PCA(n_components=2)
pca_scores = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=pca_scores, columns=["PC1", "PC2"])
pca_df["Class"] = y.values
fig = px.scatter(pca_df, x="PC1", y="PC2", color="Class", title="PCA Score Plot")
st.plotly_chart(fig, use_container_width=True)

# Classification Model
st.subheader("3. Halal vs Haram Classification")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.markdown("**Classification Report:**")
st.text(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
st.markdown("**Confusion Matrix:**")
st.dataframe(pd.DataFrame(conf_matrix, index=model.classes_, columns=model.classes_))

st.success("Prototype completed. Further modules (PLS-DA, VIP score, unknown prediction) can be added.")
