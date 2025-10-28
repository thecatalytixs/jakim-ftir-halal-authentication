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
    np.fill_diagonal(P, 0.0)

    R_off = R.copy()
    np.fill_diagonal(R_off, 0.0)
    r2_sum = np.sum(R_off**2)
    p2_sum = np.sum(P**2)
    kmo_overall = r2_sum / (r2_sum + p2_sum)

    r2_i = np.sum(R_off**2, axis=0)
    p2_i = np.sum(P**2, axis=0)
    msa = r2_i / (r2_i + p2_i)
    return float(kmo_overall), pd.Series(msa)

def sample_features_for_kmo(df_features, n_sample=1000, random_state=42):
    """Randomly sample features for approximate KMO"""
    if df_features.shape[1] <= n_sample:
        return df_features
    return df_features.sample(n=n_sample, axis=1, random_state=random_state)

# Decide computation mode
n_features = X.shape[1]
if n_features > 2000:
    st.warning(f"Dataset has {n_features} features. Using **approximate KMO** (random 1000 features).")
    X_for_kmo = sample_features_for_kmo(X[feature_cols], n_sample=1000)
    with st.spinner("Computing approximate KMO…"):
        kmo_value, msa_series = kmo_numpy(X_for_kmo.values)
else:
    X_for_kmo = X[feature_cols]
    with st.spinner("Computing exact KMO…"):
        kmo_value, msa_series = kmo_numpy(X_for_kmo.values)

def kmo_note(k):
    if k >= 0.90: return "Marvelous"
    if k >= 0.80: return "Meritorious"
    if k >= 0.70: return "Middling"
    if k >= 0.60: return "Mediocre"
    if k >= 0.50: return "Miserable"
    return "Unacceptable"

st.markdown(f"**Overall KMO:** {kmo_value:.3f}  ·  _{kmo_note(kmo_value)}_")
st.caption("KMO is computed on a subset for feasibility. PCA and PLS-DA always use **all features**.")

msa_df = msa_series.reset_index()
msa_df.columns = ["Variable", "MSA"]
st.dataframe(msa_df.sort_values("MSA", ascending=False), use_container_width=True)

# -----------------------------
# 3. PCA
# -----------------------------
st.subheader("3. Principal Component Analysis PCA")
n_pc = st.slider("Number of PCs", min_value=2, max_value=10, value=3, step=1)

scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X)

pca = PCA(n_components=n_pc, random_state=42)
pca_scores_all = pca.fit_transform(X_all_scaled)

pca_cols = [f"PC{i+1}" for i in range(n_pc)]
pca_df = pd.DataFrame(pca_scores_all, columns=pca_cols)
pca_df["Class"] = label_encoder.inverse_transform(y_encoded)
pca_df["SampleID"] = df["SampleID"].values

show_labels = st.checkbox("Show SampleID labels on PCA plot", value=False)
if n_pc >= 3:
    fig = px.scatter_3d(
        pca_df,
        x=pca_cols[0], y=pca_cols[1], z=pca_cols[2],
        color="Class",
        text="SampleID" if show_labels else None,
        title="PCA Score Plot 3D"
    )
else:
    fig = px.scatter(
        pca_df,
        x=pca_cols[0], y=pca_cols[1],
        color="Class",
        text="SampleID" if show_labels else None,
        title="PCA Score Plot 2D"
    )
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 4. PCA loadings
# -----------------------------
st.subheader("4. Variable Plot PCA Loadings")
loadings = pd.DataFrame(pca.components_.T, columns=pca_cols, index=feature_cols)
if n_pc >= 3:
    fig_loadings = px.scatter_3d(
        loadings.reset_index(), x=pca_cols[0], y=pca_cols[1], z=pca_cols[2],
        text="index", title="PCA Loadings Plot"
    )
else:
    fig_loadings = px.scatter(
        loadings.reset_index(), x=pca_cols[0], y=pca_cols[1],
        text="index", title="PCA Loadings Plot"
    )
st.plotly_chart(fig_loadings, use_container_width=True)

# -----------------------------
# 5. PCA biplot
# -----------------------------
st.subheader("5. PCA Biplot")
fig_biplot = go.Figure()
for label in pca_df["Class"].unique():
    subset = pca_df[pca_df["Class"] == label]
    mode = 'markers+text' if show_labels else 'markers'
    if n_pc >= 3:
        fig_biplot.add_trace(go.Scatter3d(
            x=subset[pca_cols[0]], y=subset[pca_cols[1]], z=subset[pca_cols[2]],
            mode=mode,
            text=subset["SampleID"] if show_labels else None,
            name=label
        ))
    else:
        fig_biplot.add_trace(go.Scatter(
            x=subset[pca_cols[0]], y=subset[pca_cols[1]],
            mode=mode,
            text=subset["SampleID"] if show_labels else None,
            name=label
        ))
st.plotly_chart(fig_biplot, use_container_width=True)

# -----------------------------
# 6. PLS scores plot
# -----------------------------
st.subheader("6. PLS scores plot for visual inspection")
n_pls_vis = st.slider("Number of PLS components for plotting", min_value=2, max_value=10, value=3, step=1)

pls_vis = PLSRegression(n_components=n_pls_vis)
pls_vis.fit(X_all_scaled, y_encoded)
pls_scores_all = pls_vis.transform(X_all_scaled)
pls_cols = [f"PLS{i+1}" for i in range(n_pls_vis)]
pls_df = pd.DataFrame(pls_scores_all, columns=pls_cols)
pls_df["Class"] = label_encoder.inverse_transform(y_encoded)
pls_df["SampleID"] = df["SampleID"].values

show_pls_labels = st.checkbox("Show SampleID labels on PLS plot", value=False)
if n_pls_vis >= 3:
    fig_pls = px.scatter_3d(
        pls_df, x=pls_cols[0], y=pls_cols[1], z=pls_cols[2],
        color="Class", text="SampleID" if show_pls_labels else None,
        title="PLS Observation Plot scores"
    )
else:
    fig_pls = px.scatter(
        pls_df, x=pls_cols[0], y=pls_cols[1],
        color="Class", text="SampleID" if show_pls_labels else None,
        title="PLS Observation Plot scores"
    )
st.plotly_chart(fig_pls, use_container_width=True)

# -----------------------------
# 7. PLS-DA with LOOCV
# -----------------------------
st.subheader("7. PLS DA classification with Leave One Out Cross Validation")

n_pls = st.slider("Number of PLS components for classification", min_value=2, max_value=10, value=3, step=1)

loo = LeaveOneOut()
X_np = X.values
y_np = y_encoded.copy()

y_true, y_pred = [], []
for train_idx, test_idx in loo.split(X_np):
    X_train, X_test = X_np[train_idx], X_np[test_idx]
    y_train, y_test = y_np[train_idx], y_np[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pls = PLSRegression(n_components=n_pls)
    pls.fit(X_train_scaled, y_train)

    X_train_scores = pls.transform(X_train_scaled)
    X_test_scores = pls.transform(X_test_scaled)

    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(X_train_scores, y_train)

    y_hat = clf.predict(X_test_scores)[0]
    y_true.append(y_test[0])
    y_pred.append(y_hat)

y_true_labels = label_encoder.inverse_transform(y_true)
y_pred_labels = label_encoder.inverse_transform(y_pred)

acc_loo = accuracy_score(y_true_labels, y_pred_labels)
st.markdown(f"**LOOCV PLS DA Accuracy:** {acc_loo:.3f}")

conf_loo = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_)
st.markdown("**Confusion Matrix LOOCV:**")
st.dataframe(pd.DataFrame(conf_loo, index=label_encoder.classes_, columns=label_encoder.classes_))

# -----------------------------
# 8. VIP Scores
# -----------------------------
st.subheader("8. VIP Scores from PLS on all data for exploration")
pls_vip = PLSRegression(n_components=min(n_pls_vis, len(feature_cols)))
pls_vip.fit(X_all_scaled, y_encoded)

T = pls_vip.x_scores_
W = pls_vip.x_weights_
Q = pls_vip.y_loadings_
p, h = W.shape
SStotal = np.sum(T**2, axis=0) * np.sum(Q**2, axis=1)
vip = np.sqrt(p * np.sum((W**2) * SStotal.reshape(1, -1), axis=1) / np.sum(SStotal))

vip_df = pd.DataFrame({'Variable': feature_cols, 'VIP_Score': vip}).sort_values(by='VIP_Score', ascending=False)
fig_vip = px.bar(vip_df.head(20), x='Variable', y='VIP_Score', title='Top 20 VIP Scores exploratory')
st.plotly_chart(fig_vip, use_container_width=True)
st.dataframe(vip_df, use_container_width=True)

# -----------------------------
# 9. External prediction
# -----------------------------
st.subheader("9. Predict classes for an external testing dataset")
st.caption("The model is trained on the entire dataset currently loaded above using the selected number of PLS components for classification")

test_file = st.file_uploader("Upload external testing dataset CSV only", type=["csv"], key="external_test")

if test_file is not None:
    test_df_raw = pd.read_csv(test_file)
    st.markdown("**Preview of testing dataset**")
    st.dataframe(test_df_raw.head(), use_container_width=True)

    if "SampleID" not in test_df_raw.columns:
        st.warning("SampleID column not found in the testing file. A sequential SampleID will be created")
        test_df_raw.insert(0, "SampleID", [f"T{i+1:03d}" for i in range(len(test_df_raw))])

    missing_feats = [c for c in feature_cols if c not in test_df_raw.columns]
    if len(missing_feats) > 0:
        st.error(f"Testing file missing {len(missing_feats)} variables (e.g. {missing_feats[:5]}).")
        st.stop()

    X_test_pred = test_df_raw[feature_cols].copy()
    for c in feature_cols:
        X_test_pred[c] = pd.to_numeric(X_test_pred[c], errors="coerce")
    X_test_pred = X_test_pred.fillna(X[feature_cols].mean(axis=0))

    scaler_final = StandardScaler()
    X_all_scaled_final = scaler_final.fit_transform(X.values)

    pls_final = PLSRegression(n_components=n_pls)
    pls_final.fit(X_all_scaled_final, y_encoded)
    X_all_pls_final = pls_final.transform(X_all_scaled_final)

    clf_final = LogisticRegression(max_iter=500, solver="lbfgs")
    clf_final.fit(X_all_pls_final, y_encoded)

    X_test_scaled_final = scaler_final.transform(X_test_pred.values)
    X_test_pls_final = pls_final.transform(X_test_scaled_final)

    y_pred_labels_final = clf_final.predict(X_test_pls_final)
    y_pred_names_final = label_encoder.inverse_transform(y_pred_labels_final)

    proba = clf_final.predict_proba(X_test_pls_final)
    proba_df = pd.DataFrame(proba, columns=[f"Prob_{c}" for c in label_encoder.classes_])

    out_df = pd.concat(
        [test_df_raw[["SampleID"]].reset_index(drop=True),
         pd.DataFrame({"Predicted_Class": y_pred_names_final}),
         proba_df],
        axis=1
    )

    st.markdown("**Predictions for testing dataset**")
    st.dataframe(out_df, use_container_width=True)
    st.download_button("Download predictions CSV", out_df.to_csv(index=False).encode("utf-8"),
                       file_name="external_predictions_plsda.csv", mime="text/csv")
