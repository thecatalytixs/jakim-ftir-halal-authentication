import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.cross_decomposition import PLSRegression
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import os

# Matplotlib light theme
plt.style.use('default')

st.set_page_config(page_title="JAKIM FTIR Halal Authentication Platform", layout="wide")
st.title("JAKIM FTIR Halal Authentication Platform")

st.caption("Upload a CSV with columns: SampleID, Class, and spectral variables (e.g., 4000..400 cm⁻¹).")

# -----------------------------
# Data loading
# -----------------------------
uploaded_file = st.file_uploader("Upload your FTIR dataset (CSV format only)", type=["csv"])

@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    # Fallbacks
    demo_path = "simulated_ftir_data.csv"
    if os.path.exists(demo_path):
        return pd.read_csv(demo_path)
    # Generate a small, realistic demo dataset if no file and no demo exists
    rng = np.random.default_rng(42)
    n_per_class = 40
    wnums = np.arange(4000, 650, -10)  # 4000 to 700 cm-1 step 10 (approx)
    p = len(wnums)

    def make_block(mu_shift):
        base = rng.normal(0, 1, size=(n_per_class, p))
        # Add a few class-specific absorbance bumps
        bumps_idx = rng.choice(np.arange(50, p-50), size=4, replace=False)
        for b in bumps_idx:
            base[:, b-3:b+3] += mu_shift
        return base

    X_halal = make_block(1.2)
    X_nonhalal = make_block(-1.0)
    X_mix = make_block(0.4)

    X = np.vstack([X_halal, X_nonhalal, X_mix])
    classes = (["Halal"] * n_per_class) + (["Non-Halal"] * n_per_class) + (["Suspect"] * n_per_class)
    sample_ids = [f"S{i+1:03d}" for i in range(len(X))]

    df_demo = pd.DataFrame(X, columns=[f"{w}" for w in wnums])
    df_demo.insert(0, "SampleID", sample_ids)
    df_demo.insert(1, "Class", classes)
    return df_demo

df = load_data(uploaded_file)

st.subheader("1. Preview of Uploaded Dataset")
st.dataframe(df.head(), use_container_width=True)

# Basic input validation
required_cols = {"SampleID", "Class"}
if not required_cols.issubset(df.columns):
    st.error("Dataset must include 'SampleID' and 'Class' columns.")
    st.stop()

feature_cols = [c for c in df.columns if c not in ["SampleID", "Class"]]
if len(feature_cols) < 3:
    st.error("Not enough spectral variables found. Include FTIR variables as numeric columns.")
    st.stop()

X = df[feature_cols].copy()
y = df["Class"].copy()

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -----------------------------
# Train test split BEFORE scaling to avoid leakage
# -----------------------------
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y_encoded, np.arange(len(X)), test_size=0.2, random_state=42, stratify=y_encoded
)

# Standardise using only training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# PCA section
# -----------------------------
st.subheader("2. Principal Component Analysis (PCA)")

n_pc = st.slider("Number of PCs to compute", min_value=2, max_value=10, value=3, step=1)

pca = PCA(n_components=n_pc, random_state=42)
pca_scores_train = pca.fit_transform(X_train_scaled)
pca_scores_all = pca.transform(scaler.transform(X))  # for plotting all points consistently

pca_cols = [f"PC{i+1}" for i in range(n_pc)]
pca_df = pd.DataFrame(pca_scores_all, columns=pca_cols)
pca_df["Class"] = label_encoder.inverse_transform(y_encoded)
pca_df["SampleID"] = df["SampleID"].values

show_labels = st.checkbox("Show SampleID labels on PCA plot", value=False)
if n_pc >= 3:
    fig = px.scatter_3d(pca_df, x=pca_cols[0], y=pca_cols[1], z=pca_cols[2], color="Class",
                        text="SampleID" if show_labels else None,
                        title="PCA Score Plot (3D)")
    if show_labels:
        fig.update_traces(textposition='top center')
else:
    fig = px.scatter(pca_df, x=pca_cols[0], y=pca_cols[1], color="Class",
                     text="SampleID" if show_labels else None,
                     title="PCA Score Plot (2D)")
st.plotly_chart(fig, use_container_width=True)

# PCA Loadings
st.subheader("3. Variable Plot (PCA Loadings)")
loadings = pd.DataFrame(pca.components_.T, columns=pca_cols, index=feature_cols)
if n_pc >= 3:
    fig_loadings = px.scatter_3d(loadings.reset_index(), x=pca_cols[0], y=pca_cols[1], z=pca_cols[2],
                                 text="index", title="PCA Loadings Plot")
else:
    fig_loadings = px.scatter(loadings.reset_index(), x=pca_cols[0], y=pca_cols[1],
                              text="index", title="PCA Loadings Plot")
st.plotly_chart(fig_loadings, use_container_width=True)

# PCA Biplot
st.subheader("4. PCA Biplot")
fig_biplot = go.Figure()

# Add scores by class
for label in pca_df["Class"].unique():
    subset = pca_df[pca_df["Class"] == label]
    mode = 'markers+text' if show_labels else 'markers'
    if n_pc >= 3:
        fig_biplot.add_trace(go.Scatter3d(
            x=subset[pca_cols[0]], y=subset[pca_cols[1]], z=subset[pca_cols[2]],
            mode=mode, text=subset["SampleID"] if show_labels else None, name=label
        ))
    else:
        fig_biplot.add_trace(go.Scatter(
            x=subset[pca_cols[0]], y=subset[pca_cols[1]],
            mode=mode, text=subset["SampleID"] if show_labels else None, name=label
        ))

# Add loadings as arrows
scale = 3.0
for i in range(loadings.shape[0]):
    if n_pc >= 3:
        fig_biplot.add_trace(go.Scatter3d(
            x=[0, loadings.iloc[i, 0]*scale],
            y=[0, loadings.iloc[i, 1]*scale],
            z=[0, loadings.iloc[i, 2]*scale if n_pc >= 3 else 0],
            mode='lines+text',
            text=["", loadings.index[i]],
            name=loadings.index[i],
            line=dict(width=2)
        ))
    else:
        fig_biplot.add_trace(go.Scatter(
            x=[0, loadings.iloc[i, 0]*scale],
            y=[0, loadings.iloc[i, 1]*scale],
            mode='lines+text',
            text=["", loadings.index[i]],
            name=loadings.index[i],
            line=dict(width=2)
        ))

fig_biplot.update_layout(
    title="PCA Biplot",
    scene=dict(xaxis_title=pca_cols[0], yaxis_title=pca_cols[1], zaxis_title=pca_cols[2] if n_pc >= 3 else None)
)
st.plotly_chart(fig_biplot, use_container_width=True)

# -----------------------------
# PLS-DA: PLS for features + Logistic Regression for classification
# -----------------------------
st.subheader("5. PLS-DA Observation Plot and Classification")

n_pls = st.slider("Number of PLS components", min_value=2, max_value=10, value=3, step=1)
pls = PLSRegression(n_components=n_pls)
pls.fit(X_train_scaled, y_train)  # supervised projection on the training set

# Scores for plotting all data consistently
X_all_scaled = scaler.transform(X)
pls_scores_all = pls.transform(X_all_scaled)  # shape: n_samples x n_pls
pls_cols = [f"PLS{i+1}" for i in range(n_pls)]
pls_df = pd.DataFrame(pls_scores_all, columns=pls_cols)
pls_df["Class"] = label_encoder.inverse_transform(y_encoded)
pls_df["SampleID"] = df["SampleID"].values

show_pls_labels = st.checkbox("Show SampleID labels on PLS-DA plot", value=False)
if n_pls >= 3:
    fig_pls = px.scatter_3d(pls_df, x=pls_cols[0], y=pls_cols[1], z=pls_cols[2],
                            color="Class", text="SampleID" if show_pls_labels else None,
                            title="PLS-DA Observation Plot (scores)")
else:
    fig_pls = px.scatter(pls_df, x=pls_cols[0], y=pls_cols[1],
                         color="Class", text="SampleID" if show_pls_labels else None,
                         title="PLS-DA Observation Plot (scores)")
st.plotly_chart(fig_pls, use_container_width=True)

# Train classifier on PLS scores from TRAIN only
pls_scores_train = pls.transform(X_train_scaled)
pls_scores_test = pls.transform(X_test_scaled)

clf = LogisticRegression(max_iter=200, multi_class="auto", solver="lbfgs", n_jobs=None)
clf.fit(pls_scores_train, y_train)

y_pred_test = clf.predict(pls_scores_test)
acc = accuracy_score(y_test, y_pred_test)
st.markdown(f"**PLS-DA Test Accuracy:** {acc:.3f}")

report_dict_test = classification_report(
    label_encoder.inverse_transform(y_test),
    label_encoder.inverse_transform(y_pred_test),
    output_dict=True
)
report_df_test = pd.DataFrame(report_dict_test).transpose().round(3)
st.markdown("**Classification Report (Test Set, PLS-DA):**")
st.dataframe(report_df_test, use_container_width=True)

conf_test = confusion_matrix(
    label_encoder.inverse_transform(y_test),
    label_encoder.inverse_transform(y_pred_test),
    labels=label_encoder.classes_
)
st.markdown("**Confusion Matrix (Test Set, PLS-DA):**")
st.dataframe(pd.DataFrame(conf_test, index=label_encoder.classes_, columns=label_encoder.classes_))

# -----------------------------
# VIP scores from PLS (x-weights, x-scores, y-loadings)
# -----------------------------
st.subheader("6. VIP Scores")
T = pls.x_scores_            # (n_train x n_pls) but computed on train fit
W = pls.x_weights_           # (p x n_pls)
Q = pls.y_loadings_          # (n_pls x 1) for single-target; for encoded classes, still works numerically

p, h = W.shape
# Sum of squares explained per component (training-based)
SStotal = np.sum(T**2, axis=0) * np.sum(Q**2, axis=1)
vip = np.sqrt(p * np.sum((W**2) * SStotal.reshape(1, -1), axis=1) / np.sum(SStotal))

vip_df = pd.DataFrame({'Variable': feature_cols, 'VIP_Score': vip}).sort_values(by='VIP_Score', ascending=False)

fig_vip = px.bar(vip_df.head(20), x='Variable', y='VIP_Score', title='Top 20 VIP Scores')
st.plotly_chart(fig_vip, use_container_width=True)
st.dataframe(vip_df, use_container_width=True)

csv_vip = vip_df.to_csv(index=False).encode('utf-8')
st.download_button("Download VIP scores (CSV)", csv_vip, file_name="vip_scores.csv", mime="text/csv")

# -----------------------------
# Baseline Logistic Regression on raw scaled features (leak-free)
# -----------------------------
st.subheader("7. Logistic Regression Baseline (on scaled features)")
base_clf = LogisticRegression(max_iter=500, multi_class="auto", solver="lbfgs")
base_clf.fit(X_train_scaled, y_train)
y_base = base_clf.predict(X_test_scaled)

report_base = classification_report(
    label_encoder.inverse_transform(y_test),
    label_encoder.inverse_transform(y_base),
    output_dict=True
)
report_base_df = pd.DataFrame(report_base).transpose().round(3)
st.dataframe(report_base_df, use_container_width=True)

conf_base = confusion_matrix(
    label_encoder.inverse_transform(y_test),
    label_encoder.inverse_transform(y_base),
    labels=label_encoder.classes_
)
st.markdown("**Confusion Matrix (Test Set, Baseline):**")
st.dataframe(pd.DataFrame(conf_base, index=label_encoder.classes_, columns=label_encoder.classes_))

# -----------------------------
# Notes
# -----------------------------
with st.expander("Notes and good practice"):
    st.write(
        """
        1. Scaling and model fitting now use only the training data to prevent leakage.
        2. PLS-DA is implemented as PLS feature extraction with a logistic classifier trained on PLS scores.
        3. VIP scores are derived from the fitted PLS model on the training set.
        4. The app auto-generates a realistic demo dataset if no CSV is provided.
        5. Consider exporting your figures and reports when preparing documentation or audit trails.
        """
    )
