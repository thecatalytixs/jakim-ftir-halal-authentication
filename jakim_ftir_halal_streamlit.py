import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cross_decomposition import PLSRegression
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

plt.style.use('default')

st.set_page_config(page_title="JAKIM FTIR Halal Authentication Platform", layout="wide")
st.title("JAKIM FTIR Halal Authentication Platform")
st.caption("Upload a CSV with columns SampleID, Class and spectral variables such as 4000 to 400 cm⁻¹")

uploaded_file = st.file_uploader("Upload your FTIR dataset CSV only", type=["csv"])

@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    demo_path = "simulated_ftir_data.csv"
    if os.path.exists(demo_path):
        return pd.read_csv(demo_path)

    rng = np.random.default_rng(42)
    n_per_class = 40
    wnums = np.arange(4000, 650, -10)

    def make_block(mu_shift):
        base = rng.normal(0, 1, size=(n_per_class, len(wnums)))
        bumps_idx = rng.choice(np.arange(50, len(wnums) - 50), size=4, replace=False)
        for b in bumps_idx:
            base[:, b - 3:b + 3] += mu_shift
        return base

    X_bovine = make_block(1.2)
    X_porcine = make_block(-1.0)
    X_fish = make_block(0.4)

    X = np.vstack([X_bovine, X_porcine, X_fish])
    classes = (["Bovine"] * n_per_class) + (["Porcine"] * n_per_class) + (["Fish"] * n_per_class)
    sample_ids = [f"S{i+1:03d}" for i in range(len(X))]

    df_demo = pd.DataFrame(X, columns=[f"{w}" for w in wnums])
    df_demo.insert(0, "SampleID", sample_ids)
    df_demo.insert(1, "Class", classes)
    return df_demo

df = load_data(uploaded_file)

# 1. Preview
st.subheader("1. Preview of Uploaded Dataset")
st.dataframe(df.head(), use_container_width=True)

required_cols = {"SampleID", "Class"}
if not required_cols.issubset(df.columns):
    st.error("Dataset must include SampleID and Class columns")
    st.stop()

feature_cols = [c for c in df.columns if c not in ["SampleID", "Class"]]
if len(feature_cols) < 3:
    st.error("Not enough spectral variables found")
    st.stop()

X = df[feature_cols].copy()
y = df["Class"].copy()
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -----------------------------
# 2. Scalable KMO
# -----------------------------
st.subheader("2. Kaiser Meyer Olkin (KMO) test (scalable for FTIR)")

def kmo_numpy(X_feat: np.ndarray):
    """Lightweight KMO using correlation and pseudo-inverse"""
    Z = (X_feat - X_feat.mean(axis=0)) / X_feat.std(axis=0, ddof=0)
    R = np.corrcoef(Z, rowvar=False)
    eps = 1e-6
    R += np.eye(R.shape[0]) * eps
    invR = np.linalg.pinv(R)
    d = np.sqrt(np.diag(invR))
    P = -invR / np.outer(d, d)
    np.fill
