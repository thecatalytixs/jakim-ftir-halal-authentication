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
import io

# Force matplotlib to use light theme
plt.style.use('default')

st.set_page_config(page_title="FTIR-Based Halal Authentication", layout="wide")
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

# Matplotlib PNG Export for 2D plot
fig_pca, ax = plt.subplots()
for label in pca_df["Class"].unique():
    subset = pca_df[pca_df["Class"] == label]
    ax.scatter(subset["PC1"], subset["PC2"], label=label)
ax.set_title("PCA Score Plot (PC1 vs PC2)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
buf = io.BytesIO()
fig_pca.savefig(buf, format="png", facecolor='white')
st.download_button("Download PCA Plot as PNG", data=buf.getvalue(), file_name="pca_plot.png", mime="image/png")

# PCA Loadings (Variable Plot)
st.subheader("3. Variable Plot (PCA Loadings)")
loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2", "PC3"], index=X.columns)
fig_loadings, ax_load = plt.subplots()
ax_load.scatter(loadings["PC1"], loadings["PC2"])
for i, txt in enumerate(loadings.index):
    ax_load.annotate(txt, (loadings["PC1"][i], loadings["PC2"][i]))
ax_load.set_title("PCA Variable Plot (Loadings PC1 vs PC2)")
ax_load.set_xlabel("PC1")
ax_load.set_ylabel("PC2")
buf_load = io.BytesIO()
fig_loadings.savefig(buf_load, format="png", facecolor='white')
st.pyplot(fig_loadings)
st.download_button("Download Variable Plot as PNG", data=buf_load.getvalue(), file_name="variable_plot.png", mime="image/png")

# PCA Biplot
st.subheader("4. PCA Biplot")
fig_biplot, ax_bi = plt.subplots()
for label in pca_df["Class"].unique():
    filtered = pca_df[pca_df["Class"] == label]
    ax_bi.scatter(filtered["PC1"], filtered["PC2"], label=label)
    if show_labels:
        for i in range(len(filtered)):
            ax_bi.annotate(filtered["SampleID"].iloc[i], (filtered["PC1"].iloc[i], filtered["PC2"].iloc[i]))
for i in range(loadings.shape[0]):
    ax_bi.arrow(0, 0, loadings.iloc[i, 0]*5, loadings.iloc[i, 1]*5, color='r', alpha=0.5)
    ax_bi.text(loadings.iloc[i, 0]*5, loadings.iloc[i, 1]*5, loadings.index[i], color='r')
ax_bi.set_title("PCA Biplot (PC1 vs PC2)")
ax_bi.set_xlabel("PC1")
ax_bi.set_ylabel("PC2")
ax_bi.legend()
buf_biplot = io.BytesIO()
fig_biplot.savefig(buf_biplot, format="png", facecolor='white')
st.pyplot(fig_biplot)
st.download_button("Download PCA Biplot as PNG", data=buf_biplot.getvalue(), file_name="pca_biplot.png", mime="image/png")
